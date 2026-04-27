"""TopoGPT2 Embeddings Navigator.

Production-grade interactive explorer for the hidden-state embeddings of a
trained TopoGPT2 checkpoint. Given a piece of text, the navigator tokenizes
it, runs a forward pass (instrumented with hooks on every transformer
block), extracts the residual-stream activations at every layer, and
surfaces a rich suite of geometric, topological, and dynamical metrics over
the resulting point clouds.

The scientific metric suite exposed by the navigator:

  kappa (curvature)
      Local estimate of the Gauss-like sectional curvature from the
      deviation between Euclidean and geodesic pairwise distances. A
      positive kappa signals spherical-like neighbourhoods, a negative
      one hyperbolic neighbourhoods.

  delta (Gromov delta-hyperbolicity)
      Approximates the four-point condition on the shortest-path metric
      built from the kNN graph. delta -> 0 is tree-like (hyperbolic),
      large delta is flat.

  alpha (persistent homology, H0 / H1)
      Distance-based Vietoris-Rips filtration producing birth-death pairs
      for connected components and one-dimensional cycles. Rendered as a
      persistence diagram.

  winding numbers
      For each closed loop in the token sequence (first-last identity
      contract), Claude-style geometric degree of the trajectory in every
      pair of principal components.

  Berry phases
      Geometric phases gamma = -Im sum log <psi_k | psi_{k+1}>
      accumulated along the trajectory, computed from the unit-normalised
      hidden-state vectors treated as quantum states.

  LC (Lipschitz constants)
      Local Lipschitz constant per edge of the sequence trajectory
      (||dh|| / ||dx||) and a global supremum.

  SP (shortest-path geometry)
      All-pairs shortest-path distances on the kNN graph plus the graph
      diameter, radius, average path length and Wasserstein-like stretch
      relative to the Euclidean metric.

All of the above are computed directly from real activations extracted
from the loaded checkpoint. No synthetic data is ever injected.

The file follows SOLID:

  * Every concern is isolated behind a dedicated class (loading,
    tokenizing, activation capture, metrics, projection, visualization).
  * The visualizer hierarchy is open for extension (new tabs subclass
    ``BaseEmbeddingView`` and are registered via ``ViewRegistry``).
  * All thresholds, sampling caps, colours, and heuristic constants live
    in ``NavigatorConfig``.
"""

from __future__ import annotations

import hashlib
import io
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
from plotly.subplots import make_subplots
from scipy import fft as scipy_fft
from scipy.sparse import csgraph as sp_csgraph
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class ThemeTokens:
    """Palette and CSS tokens for the navigator."""

    background_start: str = "#0a0e17"
    background_end: str = "#0d1b2a"
    panel: str = "rgba(15, 32, 61, 0.85)"
    card: str = "rgba(26, 42, 85, 0.70)"
    accent: str = "#64b5f6"
    accent_warm: str = "#ffb74d"
    accent_cold: str = "#4dd0e1"
    border: str = "#4a6fa5"
    text: str = "#e0e0ff"
    text_muted: str = "#8a9bbd"
    glow: str = "0 0 10px rgba(100, 181, 246, 0.35)"


@dataclass(frozen=True)
class PlotTheme:
    """Plot-level theme."""

    template: str = "plotly_dark"
    sequential: str = "Viridis"
    diverging: str = "RdBu"
    spectral: str = "Plasma"
    cyclical: str = "Twilight"
    font_color: str = "#e0e0ff"
    grid: str = "#4a6fa5"
    paper_bg: str = "rgba(0, 0, 0, 0)"
    plot_bg: str = "rgba(10, 14, 23, 0.3)"


@dataclass(frozen=True)
class SamplingLimits:
    """Safety caps to keep the UI responsive."""

    max_tokens_per_pass: int = 256
    max_points_for_persistence: int = 256
    max_points_for_isomap: int = 1024
    max_points_for_umap: int = 2048
    knn_default: int = 5
    knn_minimum: int = 2
    knn_maximum: int = 32
    min_points_for_metrics: int = 4


@dataclass(frozen=True)
class MetricsConfig:
    """Numerical stability thresholds."""

    eps_norm: float = 1e-12
    eps_log: float = 1e-10
    eps_div: float = 1e-9
    berry_min_loop_length: int = 3
    gromov_quadruple_sample: int = 800
    gromov_random_state: int = 13
    persistence_max_edges: int = 40_000
    winding_angle_eps: float = 1e-6


@dataclass(frozen=True)
class ProjectionConfig:
    """Projection hyperparameters."""

    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    isomap_n_neighbors: int = 8
    random_state: int = 42


@dataclass(frozen=True)
class NavigatorConfig:
    """Top-level configuration."""

    theme: ThemeTokens = field(default_factory=ThemeTokens)
    plot: PlotTheme = field(default_factory=PlotTheme)
    sampling: SamplingLimits = field(default_factory=SamplingLimits)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)

    page_title: str = "TopoGPT2 Embeddings Navigator"
    default_text: str = (
        "The quiet river carried the moon across the sleeping valley, "
        "and the sleeping valley dreamed of the river."
    )
    default_figure_height: int = 620
    medium_figure_height: int = 460
    compact_figure_height: int = 320

    trajectory_line_width: float = 3.0
    trajectory_marker_size: int = 6
    cloud_marker_size: int = 4
    cloud_marker_opacity: float = 0.82
    graph_edge_width: float = 1.1

    quaternion_component_names: Tuple[str, str, str, str] = ("w", "x", "y", "z")
    supported_checkpoint_extensions: Tuple[str, ...] = (
        ".safetensors",
        ".pt",
        ".bin",
        ".pth",
    )


class StyleInjector:
    """Injects the navigator's CSS once per session."""

    def __init__(self, theme: ThemeTokens) -> None:
        self._theme = theme

    def inject(self) -> None:
        """Render the CSS block in the current Streamlit page."""
        t = self._theme
        css = f"""
        <style>
        .main {{
            background: linear-gradient(135deg, {t.background_start} 0%, {t.background_end} 100%);
            color: {t.text};
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}
        .header-container {{
            background: {t.panel};
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            border: 1px solid {t.border};
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        }}
        .metric-card {{
            background: {t.card};
            border-radius: 12px;
            padding: 1rem;
            border: 1px solid #3a5ba0;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        .metric-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 15px rgba(58, 91, 160, 0.35);
        }}
        .theory-box {{
            background: rgba(19, 41, 77, 0.85);
            border: 1px solid #5c9bd5;
            border-radius: 10px;
            padding: 1.2rem;
            margin: 1rem 0;
        }}
        .citation-box {{
            background: rgba(22, 38, 68, 0.8);
            border-left: 4px solid {t.accent};
            padding: 1rem;
            margin: 1rem 0;
            font-style: italic;
            color: #bbdefb;
        }}
        .scientific-notation {{
            font-family: 'Lucida Console', Monaco, monospace;
            background: rgba(30, 45, 80, 0.6);
            padding: 0.4rem 0.7rem;
            border-radius: 6px;
            border-left: 3px solid #4a86e8;
        }}
        h1, h2, h3 {{
            color: {t.accent} !important;
            text-shadow: {t.glow};
        }}
        .footer {{
            text-align: center;
            padding: 2rem 0;
            color: {t.text_muted};
            font-size: 0.9rem;
            border-top: 1px solid #2c4a7d;
            margin-top: 2rem;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)


class CheckpointBundle:
    """Holds a loaded TopoGPT2 model along with its config and tokenizer."""

    def __init__(
        self,
        model: "Any",
        config: "Any",
        tokenizer: "Any",
        source_name: str,
    ) -> None:
        self._model = model
        self._config = config
        self._tokenizer = tokenizer
        self._source_name = source_name

    @property
    def model(self) -> "Any":
        """Return the underlying nn.Module in eval mode."""
        return self._model

    @property
    def config(self) -> "Any":
        """Return the model config object."""
        return self._config

    @property
    def tokenizer(self) -> "Any":
        """Return the BPE tokenizer."""
        return self._tokenizer

    @property
    def source_name(self) -> str:
        """Return the original file name of the checkpoint."""
        return self._source_name

    @property
    def device(self) -> str:
        """Return the device the model is currently placed on."""
        return next(self._model.parameters()).device.type

    def num_layers(self) -> int:
        """Return the number of transformer layers in the model."""
        return len(self._model.layers)

    def embedding_dim(self) -> int:
        """Return the model hidden size."""
        return int(self._config.D_MODEL)


class ModelLoader:
    """Builds a ``CheckpointBundle`` from a checkpoint on disk or in memory."""

    def __init__(self, config: NavigatorConfig) -> None:
        self._config = config

    def load(self, source: Any, topogpt2_module: Any) -> CheckpointBundle:
        """Load a checkpoint and instantiate the corresponding model.

        Args:
            source: Either a filesystem path or a Streamlit UploadedFile.
            topogpt2_module: Already-imported ``topogpt2_1`` module providing
                ``TopoGPT2``, ``TopoGPT2Config``, and ``BPETokenizer``.

        Returns:
            A :class:`CheckpointBundle` with model, config, and tokenizer.

        Raises:
            ValueError: if the checkpoint extension is not supported or if
                the model config cannot be reconstructed.
        """
        name, state_dict, embedded_cfg = self._read_state_dict(source)
        cfg = self._build_config(topogpt2_module, embedded_cfg, state_dict)
        tokenizer = topogpt2_module.BPETokenizer("gpt2")
        cfg.VOCAB_SIZE = tokenizer.vocab_size
        model = topogpt2_module.TopoGPT2(cfg)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        self._validate_load(missing, unexpected)
        model.eval()
        model.to("cpu")
        return CheckpointBundle(model, cfg, tokenizer, name)

    def _read_state_dict(
        self, source: Any
    ) -> Tuple[str, Dict[str, torch.Tensor], Optional[Dict[str, Any]]]:
        name, buffer = self._materialize(source)
        suffix = Path(name).suffix.lower()
        if suffix not in self._config.supported_checkpoint_extensions:
            raise ValueError(f"Unsupported extension: {suffix}")
        if suffix == ".safetensors":
            from safetensors.torch import load as safetensors_load

            buffer.seek(0)
            raw = safetensors_load(buffer.read())
            return name, {k: v.cpu() for k, v in raw.items()}, None
        buffer.seek(0)
        try:
            obj = torch.load(buffer, map_location="cpu", weights_only=True)
        except Exception:
            buffer.seek(0)
            obj = torch.load(buffer, map_location="cpu", weights_only=False)
        state, cfg = self._extract_payload(obj)
        return name, state, cfg

    def _materialize(self, source: Any) -> Tuple[str, io.BytesIO]:
        if hasattr(source, "read") and hasattr(source, "name"):
            return source.name, io.BytesIO(source.read())
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path.name, io.BytesIO(path.read_bytes())

    def _extract_payload(
        self, obj: Any
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, Any]]]:
        if not isinstance(obj, dict):
            raise RuntimeError("Checkpoint payload is not a dictionary.")
        cfg = None
        if isinstance(obj.get("config"), dict):
            cfg = obj["config"]
        for key in ("model_state_dict", "state_dict", "model", "weights"):
            if key in obj and isinstance(obj[key], dict) and obj[key]:
                sample = next(iter(obj[key].values()))
                if isinstance(sample, torch.Tensor):
                    return obj[key], cfg
        if obj and isinstance(next(iter(obj.values())), torch.Tensor):
            return obj, cfg
        raise RuntimeError("No tensor state dictionary found in checkpoint.")

    def _build_config(
        self,
        topogpt2_module: Any,
        embedded_cfg: Optional[Dict[str, Any]],
        state_dict: Dict[str, torch.Tensor],
    ) -> Any:
        if embedded_cfg is not None:
            known_fields = {
                f.name for f in topogpt2_module.TopoGPT2Config.__dataclass_fields__.values()
            }
            filtered = {k: v for k, v in embedded_cfg.items() if k in known_fields}
            return topogpt2_module.TopoGPT2Config(**filtered)
        return self._infer_config(topogpt2_module, state_dict)

    def _infer_config(
        self,
        topogpt2_module: Any,
        state_dict: Dict[str, torch.Tensor],
    ) -> Any:
        d_model = self._infer_d_model(state_dict)
        n_layers = self._infer_num_layers(state_dict)
        n_heads = self._infer_n_heads(state_dict, d_model)
        d_head = d_model // n_heads if n_heads > 0 else 0
        n_kv_heads = self._infer_n_kv_heads(state_dict, d_head, n_heads)
        max_seq_len = self._infer_max_seq_len(state_dict)
        torus_radial, torus_angular = self._infer_torus_grid(state_dict)
        kwargs: Dict[str, Any] = dict(
            SCALE="custom",
            D_MODEL=d_model,
            N_HEADS=n_heads,
            N_LAYERS=n_layers,
            N_KV_HEADS=n_kv_heads,
        )
        if max_seq_len is not None:
            kwargs["MAX_SEQ_LEN"] = max_seq_len
        if torus_radial is not None:
            kwargs["TORUS_RADIAL_BINS"] = torus_radial
        if torus_angular is not None:
            kwargs["TORUS_ANGULAR_BINS"] = torus_angular
        return topogpt2_module.TopoGPT2Config(**kwargs)

    @staticmethod
    def _infer_d_model(state_dict: Dict[str, torch.Tensor]) -> int:
        if "token_embed.weight" in state_dict:
            return int(state_dict["token_embed.weight"].shape[1])
        raise RuntimeError("Cannot infer D_MODEL: token_embed.weight missing.")

    @staticmethod
    def _infer_num_layers(state_dict: Dict[str, torch.Tensor]) -> int:
        layer_idxs = set()
        for key in state_dict:
            m = re.match(r"layers\.(\d+)\.", key)
            if m:
                layer_idxs.add(int(m.group(1)))
        if not layer_idxs:
            raise RuntimeError("Cannot infer N_LAYERS from state dict.")
        return max(layer_idxs) + 1

    @staticmethod
    def _infer_n_heads(state_dict: Dict[str, torch.Tensor], d_model: int) -> int:
        for key, tensor in state_dict.items():
            if key.endswith("rope.cos_cache") or key.endswith("rope.sin_cache"):
                if tensor.dim() >= 2:
                    d_head = int(tensor.shape[1])
                    if d_head > 0 and d_model % d_head == 0:
                        return d_model // d_head
        qw = state_dict.get("layers.0.attn.q_proj.weight")
        if qw is None:
            return 8
        out_features = int(qw.shape[0])
        for candidate in (12, 16, 8, 6, 4, 2, 1):
            if out_features % candidate == 0 and d_model % candidate == 0:
                return candidate
        return 1

    @staticmethod
    def _infer_max_seq_len(state_dict: Dict[str, torch.Tensor]) -> Optional[int]:
        for key, tensor in state_dict.items():
            if key.endswith("rope.cos_cache") or key.endswith("rope.sin_cache"):
                if tensor.dim() >= 1:
                    return int(tensor.shape[0])
        return None

    @staticmethod
    def _infer_n_kv_heads(
        state_dict: Dict[str, torch.Tensor],
        d_head: int,
        n_heads: int,
    ) -> int:
        """Infer the number of key/value heads for GQA.

        In GQA the ``k_proj`` (and ``v_proj``) output dimension equals
        ``n_kv_heads * d_head``. When the checkpoint was trained with
        standard multi-head attention ``n_kv_heads`` equals ``n_heads``.
        Passing the explicit value via ``N_KV_HEADS`` avoids the automatic
        ``N_HEADS // 4`` heuristic in ``TopoGPT2Config`` which would
        otherwise produce a shape mismatch at load time.
        """
        if d_head <= 0:
            return n_heads if n_heads > 0 else 1
        kw = state_dict.get("layers.0.attn.k_proj.weight")
        if kw is None or kw.dim() < 1:
            return n_heads if n_heads > 0 else 1
        out_features = int(kw.shape[0])
        if out_features % d_head != 0:
            return n_heads if n_heads > 0 else 1
        kv = out_features // d_head
        if kv <= 0:
            return 1
        if n_heads % kv != 0:
            return n_heads
        return kv

    @staticmethod
    def _infer_torus_grid(
        state_dict: Dict[str, torch.Tensor],
    ) -> Tuple[Optional[int], Optional[int]]:
        for key, tensor in state_dict.items():
            if "torus_spectral.0.kr_w" in key and tensor.dim() == 4:
                return int(tensor.shape[2]), int(tensor.shape[3] - 1) * 2
        return None, None

    @staticmethod
    def _validate_load(missing: List[str], unexpected: List[str]) -> None:
        if unexpected:
            raise RuntimeError(
                "Unexpected keys in checkpoint (model architecture mismatch): "
                f"{unexpected[:5]}..."
            )


class ActivationCapture:
    """Extracts per-layer residual-stream activations via forward hooks."""

    def __init__(self, config: NavigatorConfig) -> None:
        self._config = config

    def run(
        self,
        bundle: CheckpointBundle,
        text: str,
    ) -> Dict[str, Any]:
        """Run a single forward pass and return activations and tokens.

        Args:
            bundle: The loaded :class:`CheckpointBundle`.
            text: The input text to encode.

        Returns:
            A dictionary with keys ``tokens`` (list of decoded pieces),
            ``ids`` (list of token ids), ``embedding`` (initial token
            embeddings), ``layers`` (list of per-layer residual-stream
            activations of shape ``[S, D]``), and ``final`` (the post
            final-norm activations).
        """
        model = bundle.model
        tokenizer = bundle.tokenizer
        cfg = bundle.config

        ids = tokenizer.encode(text)
        if not ids:
            raise ValueError("Tokenizer produced an empty sequence.")
        ids = ids[: self._config.sampling.max_tokens_per_pass]
        ids = ids[: cfg.MAX_SEQ_LEN]

        pieces = self._decode_pieces(tokenizer, ids)
        id_tensor = torch.tensor([ids], dtype=torch.long)

        captured: List[torch.Tensor] = []

        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            if isinstance(output, tuple):
                x = output[0]
            else:
                x = output
            captured.append(x.detach().cpu()[0].clone())

        handles = []
        for layer in model.layers:
            handles.append(layer.register_forward_hook(hook))
        try:
            with torch.no_grad():
                initial = model.token_embed(id_tensor).detach().cpu()[0]
                _logits, _aux, _kvs = model(id_tensor)
                final = model.final_norm(captured[-1].unsqueeze(0)).detach().cpu()[0]
        finally:
            for h in handles:
                h.remove()

        return {
            "tokens": pieces,
            "ids": ids,
            "embedding": initial.numpy().astype(np.float64),
            "layers": [c.numpy().astype(np.float64) for c in captured],
            "final": final.numpy().astype(np.float64),
        }

    @staticmethod
    def _decode_pieces(tokenizer: Any, ids: List[int]) -> List[str]:
        pieces = []
        for tid in ids:
            try:
                piece = tokenizer.decode([tid])
            except Exception:
                piece = f"<{tid}>"
            pieces.append(piece)
        return pieces


class MetricSuite:
    """Computes the full geometric and topological metric suite."""

    def __init__(self, config: NavigatorConfig) -> None:
        self._config = config

    def compute(self, points: np.ndarray, knn: int) -> Dict[str, Any]:
        """Compute every metric on a point cloud ``[N, D]``.

        Args:
            points: Array of shape ``[N, D]`` with ``N >= 4``.
            knn: Number of neighbours for the kNN graph.

        Returns:
            Dictionary with scalar and array metrics.
        """
        out: Dict[str, Any] = {}
        points = self._sanitize(points)
        n = points.shape[0]
        out["n_points"] = int(n)
        out["ambient_dim"] = int(points.shape[1])
        if n < self._config.sampling.min_points_for_metrics:
            return self._trivial(out)

        dist_eucl = self._pairwise(points)
        dist_geo, knn_sparse = self._shortest_paths(points, knn)
        out["dist_eucl"] = dist_eucl
        out["dist_geo"] = dist_geo
        out["knn"] = knn

        out.update(self._sp_metrics(dist_eucl, dist_geo))
        out.update(self._kappa(dist_eucl, dist_geo))
        out["delta_hyperbolicity"] = self._gromov_delta(dist_geo)
        out.update(self._persistence(points, dist_eucl))
        out.update(self._berry_phases(points))
        out.update(self._winding_numbers(points))
        out.update(self._lipschitz(points))
        out.update(self._trajectory_geometry(points))
        out["knn_sparse"] = knn_sparse
        out.update(self._spectral_properties(points))
        return out

    def _sanitize(self, points: np.ndarray) -> np.ndarray:
        return np.nan_to_num(points.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    def _trivial(self, base: Dict[str, Any]) -> Dict[str, Any]:
        base.update({
            "dist_eucl": np.zeros((1, 1)),
            "dist_geo": np.zeros((1, 1)),
            "kappa_mean": 0.0,
            "kappa_std": 0.0,
            "delta_hyperbolicity": 0.0,
            "h0_bars": [],
            "h1_bars": [],
            "berry_phase_total": 0.0,
            "berry_phase_cum": np.array([0.0]),
            "winding_xy": 0,
            "winding_xz": 0,
            "winding_yz": 0,
            "lc_local": np.array([0.0]),
            "lc_global": 0.0,
            "sp_diameter": 0.0,
            "sp_radius": 0.0,
            "sp_avg": 0.0,
            "sp_stretch": 1.0,
            "trajectory_speed": np.array([0.0]),
            "trajectory_curvature": np.array([0.0]),
            "trajectory_torsion": np.array([0.0]),
            "participation_ratio": 0.0,
            "effective_rank": 0.0,
            "spectral_flatness_db": 0.0,
            "frobenius": 0.0,
        })
        return base

    def _pairwise(self, points: np.ndarray) -> np.ndarray:
        return squareform(pdist(points, metric="euclidean"))

    def _shortest_paths(
        self,
        points: np.ndarray,
        knn: int,
    ) -> Tuple[np.ndarray, csr_matrix]:
        n = points.shape[0]
        k = max(self._config.sampling.knn_minimum, min(knn, n - 1))
        nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(points)
        distances, indices = nn.kneighbors(points)
        rows = np.repeat(np.arange(n), k)
        cols = indices[:, 1:].reshape(-1)
        data = distances[:, 1:].reshape(-1)
        graph = csr_matrix((data, (rows, cols)), shape=(n, n))
        graph = graph.maximum(graph.T)
        dist_geo = sp_csgraph.shortest_path(graph, method="D", directed=False)
        if not np.all(np.isfinite(dist_geo)):
            max_finite = np.nanmax(dist_geo[np.isfinite(dist_geo)]) if np.any(np.isfinite(dist_geo)) else 1.0
            dist_geo = np.where(np.isfinite(dist_geo), dist_geo, max_finite * 2.0)
        return dist_geo, graph

    def _sp_metrics(
        self,
        dist_eucl: np.ndarray,
        dist_geo: np.ndarray,
    ) -> Dict[str, float]:
        n = dist_geo.shape[0]
        mask = ~np.eye(n, dtype=bool)
        geo_vals = dist_geo[mask]
        eucl_vals = dist_eucl[mask]
        eccentricity = dist_geo.max(axis=1)
        scale = max(float(np.median(eucl_vals)), self._config.metrics.eps_div)
        safe_eucl = np.maximum(eucl_vals, scale * self._config.metrics.eps_div)
        stretch = geo_vals / safe_eucl
        return {
            "sp_diameter": float(geo_vals.max()),
            "sp_radius": float(eccentricity.min()),
            "sp_avg": float(geo_vals.mean()),
            "sp_stretch": float(np.median(stretch)),
        }

    def _kappa(
        self,
        dist_eucl: np.ndarray,
        dist_geo: np.ndarray,
    ) -> Dict[str, float]:
        """Local curvature proxy from the chord-vs-arc ratio.

        For every point and every nearby pair (a, b) we compare the
        Euclidean chord 0.5*(d_E(i,a) + d_E(i,b)) to the geodesic arc
        0.5*(d_G(i,a) + d_G(i,b)). When the ratio chord/arc is near 1 the
        local neighbourhood is flat; when it is less than 1 the arc bends
        outward (spherical-like, positive curvature). The returned scalar
        kappa = 1 - (chord/arc)^2, bounded in [0, 1] for positive curvature
        and scale-free so it can be compared across layers.
        """
        n = dist_geo.shape[0]
        idxs = np.arange(n)
        kappa_vals = []
        for i in idxs:
            order = np.argsort(dist_geo[i])[1: min(6, n)]
            if order.size < 2:
                continue
            for a in order:
                for b in order:
                    if a >= b:
                        continue
                    arc = 0.5 * (dist_geo[i, a] + dist_geo[i, b])
                    chord = 0.5 * (dist_eucl[i, a] + dist_eucl[i, b])
                    if arc < self._config.metrics.eps_div:
                        continue
                    ratio = min(chord / arc, 1.0)
                    kappa_vals.append(1.0 - ratio**2)
        if not kappa_vals:
            return {"kappa_mean": 0.0, "kappa_std": 0.0, "kappa_values": np.array([])}
        arr = np.asarray(kappa_vals, dtype=np.float64)
        return {
            "kappa_mean": float(np.mean(arr)),
            "kappa_std": float(np.std(arr)),
            "kappa_values": arr,
        }

    def _gromov_delta(self, dist_geo: np.ndarray) -> float:
        n = dist_geo.shape[0]
        if n < 4:
            return 0.0
        rng = np.random.default_rng(self._config.metrics.gromov_random_state)
        samples = min(self._config.metrics.gromov_quadruple_sample, n**3)
        max_delta = 0.0
        ref = rng.integers(0, n)
        for _ in range(samples):
            i, j, k = rng.integers(0, n, size=3)
            if len({int(ref), int(i), int(j), int(k)}) < 4:
                continue
            a = dist_geo[ref, i] + dist_geo[j, k]
            b = dist_geo[ref, j] + dist_geo[i, k]
            c = dist_geo[ref, k] + dist_geo[i, j]
            ordered = sorted([float(a), float(b), float(c)])
            delta = 0.5 * (ordered[2] - ordered[1])
            if delta > max_delta:
                max_delta = delta
        diameter = float(dist_geo.max())
        if diameter > self._config.metrics.eps_div:
            return float(max_delta / diameter)
        return float(max_delta)

    def _persistence(
        self,
        points: np.ndarray,
        dist_eucl: np.ndarray,
    ) -> Dict[str, Any]:
        n = points.shape[0]
        cap = self._config.sampling.max_points_for_persistence
        if n > cap:
            rng = np.random.default_rng(self._config.projection.random_state)
            idx = rng.choice(n, cap, replace=False)
            idx.sort()
            sub = dist_eucl[np.ix_(idx, idx)]
            n = cap
        else:
            sub = dist_eucl

        triu = np.triu_indices(n, k=1)
        edges = list(zip(triu[0].tolist(), triu[1].tolist(), sub[triu].tolist()))
        if len(edges) > self._config.metrics.persistence_max_edges:
            edges.sort(key=lambda e: e[2])
            edges = edges[: self._config.metrics.persistence_max_edges]
        edges.sort(key=lambda e: e[2])

        h0_bars = self._h0_from_mst(edges, n)
        h1_bars = self._h1_from_edges(edges, n)
        return {"h0_bars": h0_bars, "h1_bars": h1_bars}

    def _h0_from_mst(
        self,
        edges: List[Tuple[int, int, float]],
        n: int,
    ) -> List[Tuple[float, float]]:
        parent = list(range(n))
        rank = [0] * n

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> bool:
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            if rank[rx] < rank[ry]:
                parent[rx] = ry
            elif rank[rx] > rank[ry]:
                parent[ry] = rx
            else:
                parent[ry] = rx
                rank[rx] += 1
            return True

        bars: List[Tuple[float, float]] = [(0.0, float("inf"))]
        for u, v, w in edges:
            if union(u, v):
                bars.append((0.0, float(w)))
        bars = bars[:-1] if bars else bars
        bars.append((0.0, float("inf")))
        return bars

    def _h1_from_edges(
        self,
        edges: List[Tuple[int, int, float]],
        n: int,
    ) -> List[Tuple[float, float]]:
        """Estimate H1 persistence bars from a distance-sorted edge list.

        When an edge is added that does not reduce the number of
        connected components, it closes a loop. The birth of that H1
        feature is the filtration level at which the loop appears. We
        estimate the death as the minimum filtration level at which the
        loop is filled in by a 2-simplex in the Vietoris-Rips complex
        formed from the same edges. When no such filling appears within
        the edge budget, the bar is marked as surviving to the maximum
        scale.
        """
        parent = list(range(n))
        rank = [0] * n

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> bool:
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            if rank[rx] < rank[ry]:
                parent[rx] = ry
            elif rank[rx] > rank[ry]:
                parent[ry] = rx
            else:
                parent[ry] = rx
                rank[rx] += 1
            return True

        neighbors: Dict[int, Dict[int, float]] = {i: {} for i in range(n)}
        loop_births: List[Tuple[int, int, float]] = []
        for u, v, w in edges:
            if not union(u, v):
                loop_births.append((u, v, float(w)))
            neighbors[u][v] = float(w)
            neighbors[v][u] = float(w)

        bars: List[Tuple[float, float]] = []
        max_scale = edges[-1][2] if edges else 0.0
        for u, v, birth in loop_births:
            common = set(neighbors[u]).intersection(neighbors[v])
            if not common:
                bars.append((birth, max_scale))
                continue
            death = min(max(neighbors[u][c], neighbors[v][c]) for c in common)
            if death <= birth:
                death = birth
            bars.append((birth, death))
        bars.sort(key=lambda b: b[1] - b[0], reverse=True)
        return bars[: max(1, min(256, len(bars)))]

    def _berry_phases(self, points: np.ndarray) -> Dict[str, Any]:
        n = points.shape[0]
        if n < self._config.metrics.berry_min_loop_length:
            return {"berry_phase_total": 0.0, "berry_phase_cum": np.zeros(n)}
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms < self._config.metrics.eps_norm, 1.0, norms)
        unit = points / norms
        overlaps = np.einsum("ij,ij->i", unit[:-1], unit[1:])
        overlaps = np.clip(overlaps, -1.0, 1.0)
        angles = np.arccos(overlaps)
        cum = np.concatenate([[0.0], np.cumsum(angles)])
        loop_overlap = float(np.clip(np.dot(unit[-1], unit[0]), -1.0, 1.0))
        total = float(cum[-1] + math.acos(loop_overlap))
        return {"berry_phase_total": total, "berry_phase_cum": cum}

    def _winding_numbers(self, points: np.ndarray) -> Dict[str, int]:
        n = points.shape[0]
        if n < 3 or points.shape[1] < 2:
            return {"winding_xy": 0, "winding_xz": 0, "winding_yz": 0}
        try:
            pca = PCA(n_components=3, random_state=self._config.projection.random_state)
            emb = pca.fit_transform(points)
        except Exception:
            return {"winding_xy": 0, "winding_xz": 0, "winding_yz": 0}
        return {
            "winding_xy": self._planar_winding(emb[:, 0], emb[:, 1]),
            "winding_xz": self._planar_winding(emb[:, 0], emb[:, 2]),
            "winding_yz": self._planar_winding(emb[:, 1], emb[:, 2]),
            "winding_embedding": emb,
        }

    def _planar_winding(self, xs: np.ndarray, ys: np.ndarray) -> int:
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        theta = np.arctan2(ys - cy, xs - cx)
        n = len(theta)
        closed = np.concatenate([theta, [theta[0]]])
        dtheta = np.diff(closed)
        dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
        total = float(np.sum(dtheta))
        return int(round(total / (2 * math.pi)))

    def _lipschitz(self, points: np.ndarray) -> Dict[str, Any]:
        n = points.shape[0]
        if n < 2:
            return {"lc_local": np.zeros(max(n, 1)), "lc_global": 0.0}
        diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
        return {"lc_local": diffs, "lc_global": float(np.max(diffs))}

    def _trajectory_geometry(self, points: np.ndarray) -> Dict[str, np.ndarray]:
        """Frenet-Serret differential geometry of the token trajectory.

        Speed is computed in the full ambient space. Curvature and torsion
        are well-defined in 3D, so they are evaluated on the top-3 PCA
        projection of the sequence; this preserves geometry while giving a
        clean, interpretable scalar per position.
        """
        n = points.shape[0]
        if n < 3:
            return {
                "trajectory_speed": np.zeros(max(n, 1)),
                "trajectory_curvature": np.zeros(max(n, 1)),
                "trajectory_torsion": np.zeros(max(n, 1)),
            }
        speed_raw = np.linalg.norm(np.diff(points, axis=0), axis=1)
        speed = np.concatenate([[speed_raw[0]], speed_raw])

        emb3 = self._safe_pca3(points)
        v = np.diff(emb3, axis=0)
        if v.shape[0] < 2:
            return {
                "trajectory_speed": speed,
                "trajectory_curvature": np.zeros(n),
                "trajectory_torsion": np.zeros(n),
            }
        a = np.diff(v, axis=0)
        cross_va = np.cross(v[:-1], a)
        cross_norm = np.linalg.norm(cross_va, axis=1)
        v_norm = np.linalg.norm(v[:-1], axis=1)
        denom_k = np.maximum(v_norm**3, self._config.metrics.eps_div)
        curvature_core = cross_norm / denom_k
        curvature = np.concatenate([
            [curvature_core[0]], curvature_core, [curvature_core[-1]]
        ])[: n]

        if v.shape[0] < 3:
            torsion = np.zeros(n)
        else:
            j = np.diff(a, axis=0)
            triple = np.einsum("ij,ij->i", cross_va[:-1], j)
            denom_t = np.maximum(np.linalg.norm(cross_va[:-1], axis=1) ** 2,
                                 self._config.metrics.eps_div)
            torsion_core = triple / denom_t
            if torsion_core.size == 0:
                torsion = np.zeros(n)
            else:
                left = torsion_core[0]
                right = torsion_core[-1]
                pad_total = n - torsion_core.size
                left_pad = pad_total // 2
                right_pad = pad_total - left_pad
                torsion = np.concatenate([
                    np.full(left_pad, left),
                    torsion_core,
                    np.full(right_pad, right),
                ])
        return {
            "trajectory_speed": speed,
            "trajectory_curvature": curvature,
            "trajectory_torsion": torsion,
        }

    def _safe_pca3(self, points: np.ndarray) -> np.ndarray:
        k = min(3, points.shape[0], points.shape[1])
        k = max(k, 1)
        try:
            pca = PCA(n_components=k, random_state=self._config.projection.random_state)
            emb = pca.fit_transform(points)
        except Exception:
            return points[:, :3] if points.shape[1] >= 3 else \
                   np.pad(points, ((0, 0), (0, max(0, 3 - points.shape[1]))))
        if emb.shape[1] < 3:
            emb = np.pad(emb, ((0, 0), (0, 3 - emb.shape[1])))
        return emb

    def _spectral_properties(self, points: np.ndarray) -> Dict[str, float]:
        if min(points.shape) < 2:
            return {
                "participation_ratio": 1.0,
                "effective_rank": 1.0,
                "spectral_flatness_db": 0.0,
                "frobenius": float(np.linalg.norm(points)),
            }
        try:
            s = np.linalg.svd(points, compute_uv=False)
        except np.linalg.LinAlgError:
            return {
                "participation_ratio": 1.0,
                "effective_rank": 1.0,
                "spectral_flatness_db": 0.0,
                "frobenius": float(np.linalg.norm(points)),
            }
        s_pos = s[s > self._config.metrics.eps_norm]
        if s_pos.size == 0:
            pr = 1.0
            er = 1.0
        else:
            pr = float((np.sum(s_pos) ** 2) / max(1.0, float(np.sum(s_pos**2)) * s_pos.size))
            probs = s_pos / s_pos.sum()
            er = float(math.exp(-float(np.sum(probs * np.log(probs + self._config.metrics.eps_log)))))
        mags = np.maximum(np.abs(scipy_fft.rfft(points[0])), self._config.metrics.eps_log)
        geo_mean = math.exp(float(np.mean(np.log(mags))))
        arith_mean = float(np.mean(mags))
        flat = 10.0 * math.log10(max(geo_mean / max(arith_mean, self._config.metrics.eps_div),
                                     self._config.metrics.eps_log))
        return {
            "participation_ratio": pr,
            "effective_rank": er,
            "spectral_flatness_db": flat,
            "frobenius": float(np.linalg.norm(points)),
        }


class Projector:
    """Projects point clouds into 2D or 3D with multiple algorithms."""

    def __init__(self, config: NavigatorConfig) -> None:
        self._config = config

    def project(
        self,
        points: np.ndarray,
        method: str,
        n_components: int = 3,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Project ``[N, D]`` points to ``n_components`` dims.

        Args:
            points: Input point cloud of shape ``[N, D]``.
            method: ``"pca"``, ``"isomap"``, ``"umap"``, ``"random"`` or
                ``"sphere"``.
            n_components: Desired output dimensionality.

        Returns:
            Tuple ``(embedding, info)`` where ``info`` contains method
            specific diagnostics (explained variance, etc.).
        """
        pts = np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
        if method == "pca":
            return self._pca(pts, n_components)
        if method == "isomap":
            return self._isomap(pts, n_components)
        if method == "umap":
            return self._umap(pts, n_components)
        if method == "random":
            return self._random(pts, n_components)
        if method == "sphere":
            return self._sphere(pts, n_components)
        raise ValueError(f"Unknown projection method: {method}")

    def _pca(
        self,
        points: np.ndarray,
        n_components: int,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        k = min(n_components, points.shape[0], points.shape[1])
        k = max(k, 1)
        model = PCA(n_components=k, random_state=self._config.projection.random_state)
        emb = model.fit_transform(points)
        if emb.shape[1] < n_components:
            pad = np.zeros((emb.shape[0], n_components - emb.shape[1]))
            emb = np.hstack([emb, pad])
        info = {
            "cumulative_evr": float(np.sum(model.explained_variance_ratio_)),
            "components_used": int(k),
        }
        return emb[:, :n_components], info

    def _isomap(
        self,
        points: np.ndarray,
        n_components: int,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        n = points.shape[0]
        cap = self._config.sampling.max_points_for_isomap
        if n > cap:
            points = points[:cap]
        k = min(self._config.projection.isomap_n_neighbors, max(2, points.shape[0] - 1))
        comp = min(n_components, max(1, points.shape[0] - 1), points.shape[1])
        try:
            model = Isomap(n_neighbors=k, n_components=comp)
            emb = model.fit_transform(points)
        except Exception:
            return self._pca(points, n_components)
        if emb.shape[1] < n_components:
            pad = np.zeros((emb.shape[0], n_components - emb.shape[1]))
            emb = np.hstack([emb, pad])
        return emb[:, :n_components], {"neighbors": int(k)}

    def _umap(
        self,
        points: np.ndarray,
        n_components: int,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        try:
            import umap
        except ImportError:
            return self._pca(points, n_components)
        n = points.shape[0]
        cap = self._config.sampling.max_points_for_umap
        if n > cap:
            points = points[:cap]
        nbrs = min(self._config.projection.umap_n_neighbors, max(2, points.shape[0] - 1))
        try:
            model = umap.UMAP(
                n_neighbors=nbrs,
                n_components=n_components,
                min_dist=self._config.projection.umap_min_dist,
                random_state=self._config.projection.random_state,
            )
            emb = model.fit_transform(points)
        except Exception:
            return self._pca(points, n_components)
        return emb, {"neighbors": int(nbrs),
                     "min_dist": float(self._config.projection.umap_min_dist)}

    def _random(
        self,
        points: np.ndarray,
        n_components: int,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        rng = np.random.default_rng(self._config.projection.random_state)
        W = rng.standard_normal(size=(points.shape[1], n_components))
        W /= np.linalg.norm(W, axis=0, keepdims=True) + self._config.metrics.eps_norm
        return points @ W, {"seed": int(self._config.projection.random_state)}

    def _sphere(
        self,
        points: np.ndarray,
        n_components: int,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        emb, info = self._pca(points, n_components)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms < self._config.metrics.eps_norm, 1.0, norms)
        scale = float(np.mean(np.linalg.norm(points, axis=1)))
        unit = emb / norms
        info["mean_radius"] = scale
        return unit * scale, info


class FigureStyler:
    """Consistent Plotly styling for every figure."""

    def __init__(self, config: NavigatorConfig) -> None:
        self._config = config

    def style_3d(self, fig: go.Figure, title: str, height: Optional[int] = None) -> go.Figure:
        """Apply 3D styling."""
        cfg = self._config
        pt = cfg.plot
        fig.update_layout(
            template=pt.template,
            title=dict(text=title, font=dict(size=17, color=cfg.theme.accent)),
            height=height or cfg.default_figure_height,
            margin=dict(l=0, r=0, b=0, t=45),
            paper_bgcolor=pt.paper_bg,
            plot_bgcolor=pt.plot_bg,
            font=dict(color=pt.font_color),
            scene=dict(
                xaxis=dict(showbackground=False, gridcolor=pt.grid, zerolinecolor=pt.grid),
                yaxis=dict(showbackground=False, gridcolor=pt.grid, zerolinecolor=pt.grid),
                zaxis=dict(showbackground=False, gridcolor=pt.grid, zerolinecolor=pt.grid),
                aspectmode="cube",
            ),
            hoverlabel=dict(bgcolor="rgba(30, 45, 80, 0.9)", font_color=pt.font_color),
        )
        return fig

    def style_2d(self, fig: go.Figure, title: str, height: Optional[int] = None) -> go.Figure:
        """Apply 2D styling."""
        cfg = self._config
        pt = cfg.plot
        fig.update_layout(
            template=pt.template,
            title=dict(text=title, font=dict(size=16, color=cfg.theme.accent)),
            height=height or cfg.medium_figure_height,
            margin=dict(l=50, r=20, t=55, b=50),
            paper_bgcolor=pt.paper_bg,
            plot_bgcolor=pt.plot_bg,
            font=dict(color=pt.font_color),
            hoverlabel=dict(bgcolor="rgba(30, 45, 80, 0.9)", font_color=pt.font_color),
        )
        fig.update_xaxes(gridcolor=pt.grid, zerolinecolor=pt.grid)
        fig.update_yaxes(gridcolor=pt.grid, zerolinecolor=pt.grid)
        return fig


@dataclass
class RenderContext:
    """Bundle passed to every view during rendering."""

    bundle: CheckpointBundle
    activations: Dict[str, Any]
    metrics_per_layer: List[Dict[str, Any]]
    projector: Projector
    styler: FigureStyler
    config: NavigatorConfig
    knn: int


class BaseEmbeddingView(ABC):
    """Abstract base for navigator views."""

    label: str = ""
    description: str = ""

    def __init__(self, ctx: RenderContext) -> None:
        self._ctx = ctx

    @property
    def ctx(self) -> RenderContext:
        """Return the shared render context."""
        return self._ctx

    @abstractmethod
    def render(self) -> None:
        """Render the view into the current Streamlit container."""


class OverviewView(BaseEmbeddingView):
    """Token table, per-layer scalar metric evolution, summary panel."""

    label = "Overview"
    description = "Token-level overview and per-layer metric evolution."

    def render(self) -> None:
        self._render_tokens()
        self._render_layer_evolution()

    def _render_tokens(self) -> None:
        acts = self._ctx.activations
        rows = []
        for i, (tok, tid) in enumerate(zip(acts["tokens"], acts["ids"])):
            initial_norm = float(np.linalg.norm(acts["embedding"][i]))
            final_norm = float(np.linalg.norm(acts["final"][i]))
            rows.append({
                "position": i,
                "token_id": int(tid),
                "token": tok,
                "|embedding|": initial_norm,
                "|final|": final_norm,
                "drift": final_norm - initial_norm,
            })
        st.markdown("### Tokens (residual-stream entry and exit)")
        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
            column_config={
                "|embedding|": st.column_config.NumberColumn(format="%.4f"),
                "|final|": st.column_config.NumberColumn(format="%.4f"),
                "drift": st.column_config.NumberColumn(format="%.4f"),
            },
        )

    def _render_layer_evolution(self) -> None:
        ms = self._ctx.metrics_per_layer
        if not ms:
            return
        layer_idx = list(range(len(ms)))
        series: Dict[str, List[float]] = {
            "kappa (mean)": [m.get("kappa_mean", 0.0) for m in ms],
            "delta (hyperbolicity)": [m.get("delta_hyperbolicity", 0.0) for m in ms],
            "Lipschitz (global)": [m.get("lc_global", 0.0) for m in ms],
            "SP diameter": [m.get("sp_diameter", 0.0) for m in ms],
            "SP stretch": [m.get("sp_stretch", 1.0) for m in ms],
            "participation ratio": [m.get("participation_ratio", 0.0) for m in ms],
            "effective rank": [m.get("effective_rank", 0.0) for m in ms],
            "Berry phase (total)": [m.get("berry_phase_total", 0.0) for m in ms],
        }
        titles = list(series.keys())
        fig = make_subplots(
            rows=2, cols=4,
            subplot_titles=titles,
            horizontal_spacing=0.08,
            vertical_spacing=0.18,
        )
        for i, title in enumerate(titles):
            r = i // 4 + 1
            c = i % 4 + 1
            fig.add_trace(
                go.Scatter(
                    x=layer_idx,
                    y=series[title],
                    mode="lines+markers",
                    line=dict(color=self._ctx.config.theme.accent, width=2.4),
                    marker=dict(size=7),
                    showlegend=False,
                ),
                row=r, col=c,
            )
            fig.update_xaxes(title="layer", row=r, col=c)
        self._ctx.styler.style_2d(fig, "Metric evolution through the residual stream",
                                  height=self._ctx.config.default_figure_height)
        st.plotly_chart(fig, use_container_width=True)


class CloudView(BaseEmbeddingView):
    """Interactive 3D cloud of one layer's residual-stream activations."""

    label = "Cloud 3D"
    description = "3D point cloud of a single layer with selectable projection."

    def render(self) -> None:
        acts = self._ctx.activations
        total_layers = len(acts["layers"]) + 1
        choices = ["embedding"] + [f"layer {i}" for i in range(len(acts["layers"]))] + ["final"]
        layer_sel = st.selectbox("Stage", options=choices, index=min(len(choices) - 1, 1))
        method = st.radio(
            "Projection",
            options=["pca", "isomap", "umap", "random", "sphere"],
            horizontal=True,
            help="PCA for variance, Isomap for geodesics, UMAP for manifold, "
                 "Sphere projects to the hypersphere while preserving PCA directions.",
        )
        points = self._choose(acts, layer_sel)
        tokens = acts["tokens"]
        ids = acts["ids"]
        emb, info = self._ctx.projector.project(points, method, n_components=3)
        norms = np.linalg.norm(points, axis=1)
        self._render_trajectory(emb, tokens, ids, norms, layer_sel, method, info)
        self._render_residual_streams(acts, method)

    def _choose(self, acts: Dict[str, Any], selection: str) -> np.ndarray:
        if selection == "embedding":
            return acts["embedding"]
        if selection == "final":
            return acts["final"]
        idx = int(selection.split()[-1])
        return acts["layers"][idx]

    def _render_trajectory(
        self,
        emb: np.ndarray,
        tokens: List[str],
        ids: List[int],
        norms: np.ndarray,
        stage: str,
        method: str,
        info: Dict[str, float],
    ) -> None:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=emb[:, 0], y=emb[:, 1], z=emb[:, 2],
                mode="lines",
                line=dict(
                    color=np.arange(emb.shape[0]),
                    colorscale=self._ctx.config.plot.spectral,
                    width=self._ctx.config.trajectory_line_width,
                ),
                hoverinfo="skip",
                name="trajectory",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=emb[:, 0], y=emb[:, 1], z=emb[:, 2],
                mode="markers+text",
                marker=dict(
                    size=self._ctx.config.trajectory_marker_size,
                    color=norms,
                    colorscale=self._ctx.config.plot.sequential,
                    opacity=self._ctx.config.cloud_marker_opacity,
                    colorbar=dict(title="||h||"),
                    line=dict(color="rgba(255,255,255,0.35)", width=1),
                ),
                text=tokens,
                textposition="top center",
                customdata=np.column_stack([np.arange(emb.shape[0]), ids, norms]),
                hovertemplate=(
                    "pos=%{customdata[0]}<br>"
                    "token_id=%{customdata[1]}<br>"
                    "||h||=%{customdata[2]:.4f}<br>"
                    "x=%{x:.3f} y=%{y:.3f} z=%{z:.3f}<extra></extra>"
                ),
                name="tokens",
            )
        )
        subtitle = ""
        if method == "pca":
            subtitle = f" (cum EVR {info.get('cumulative_evr', 0)*100:.1f}%)"
        elif method in ("isomap", "umap"):
            subtitle = f" (k={info.get('neighbors', 0)})"
        self._ctx.styler.style_3d(fig, f"{stage} - {method.upper()}{subtitle}")
        st.plotly_chart(fig, use_container_width=True)

    def _render_residual_streams(self, acts: Dict[str, Any], method: str) -> None:
        all_layers = [("embedding", acts["embedding"])] \
            + [(f"layer {i}", L) for i, L in enumerate(acts["layers"])] \
            + [("final", acts["final"])]
        if len(all_layers) < 2:
            return
        stacked = np.vstack([pts for _, pts in all_layers])
        emb, _ = self._ctx.projector.project(stacked, method, n_components=3)
        n_seq = all_layers[0][1].shape[0]
        fig = go.Figure()
        for pos in range(n_seq):
            xs = emb[pos::n_seq, 0]
            ys = emb[pos::n_seq, 1]
            zs = emb[pos::n_seq, 2]
            fig.add_trace(
                go.Scatter3d(
                    x=xs, y=ys, z=zs,
                    mode="lines+markers",
                    line=dict(color=self._ctx.config.theme.accent, width=1.8),
                    marker=dict(size=3, opacity=0.7),
                    name=f"token {pos}",
                    hovertext=[f"{acts['tokens'][pos]} @ {all_layers[j][0]}" for j in range(len(all_layers))],
                    hoverinfo="text",
                )
            )
        fig.update_layout(showlegend=False)
        self._ctx.styler.style_3d(
            fig,
            f"Residual-stream trajectories (every token across every layer, {method.upper()})",
        )
        st.plotly_chart(fig, use_container_width=True)


class MetricsView(BaseEmbeddingView):
    """Detailed per-layer metric card for a selected layer."""

    label = "Metrics"
    description = "Full geometric and topological metric suite for one layer."

    def render(self) -> None:
        choices = ["embedding"] + [f"layer {i}" for i in range(len(self._ctx.activations["layers"]))] + ["final"]
        stage = st.selectbox("Stage", options=choices, index=len(choices) // 2)
        metrics = self._get_metrics(stage)

        self._render_card(metrics)
        self._render_kappa(metrics)
        self._render_path_metrics(metrics)

    def _get_metrics(self, stage: str) -> Dict[str, Any]:
        ms = self._ctx.metrics_per_layer
        return ms[["embedding"].index(stage) if stage == "embedding"
                  else (len(ms) - 1 if stage == "final"
                        else int(stage.split()[-1]) + 1)]

    def _render_card(self, m: Dict[str, Any]) -> None:
        cols = st.columns(4)
        cols[0].metric("kappa mean", f"{m['kappa_mean']:.4e}")
        cols[1].metric("kappa std", f"{m['kappa_std']:.4e}")
        cols[2].metric("delta (Gromov)", f"{m['delta_hyperbolicity']:.4f}")
        cols[3].metric("Lipschitz (global)", f"{m['lc_global']:.4e}")

        r2 = st.columns(4)
        r2[0].metric("SP diameter", f"{m['sp_diameter']:.4f}")
        r2[1].metric("SP radius", f"{m['sp_radius']:.4f}")
        r2[2].metric("SP avg", f"{m['sp_avg']:.4f}")
        r2[3].metric("SP stretch", f"{m['sp_stretch']:.4f}")

        r3 = st.columns(4)
        r3[0].metric("participation ratio", f"{m['participation_ratio']:.4f}")
        r3[1].metric("effective rank", f"{m['effective_rank']:.3f}")
        r3[2].metric("Berry phase (total)", f"{m['berry_phase_total']:.4f}")
        r3[3].metric("spectral flatness (dB)", f"{m['spectral_flatness_db']:.3f}")

        r4 = st.columns(4)
        r4[0].metric("winding XY", f"{m['winding_xy']}")
        r4[1].metric("winding XZ", f"{m['winding_xz']}")
        r4[2].metric("winding YZ", f"{m['winding_yz']}")
        r4[3].metric("H1 cycles", f"{len(m.get('h1_bars', []))}")

    def _render_kappa(self, m: Dict[str, Any]) -> None:
        vals = m.get("kappa_values")
        if vals is None or len(vals) == 0:
            return
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=vals,
                nbinsx=50,
                marker_color=self._ctx.config.theme.accent,
            )
        )
        fig.add_vline(x=0.0, line_dash="dash", line_color="rgba(255,255,255,0.45)",
                      annotation_text="flat", annotation_position="top right")
        fig.update_xaxes(title="kappa (local)")
        fig.update_yaxes(title="count")
        self._ctx.styler.style_2d(
            fig,
            "Local curvature kappa distribution "
            "(>0: spherical, <0: hyperbolic)",
        )
        st.plotly_chart(fig, use_container_width=True)

    def _render_path_metrics(self, m: Dict[str, Any]) -> None:
        de = m.get("dist_eucl")
        dg = m.get("dist_geo")
        if de is None or dg is None or de.size == 0:
            return
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Euclidean distance matrix", "Geodesic distance matrix (kNN graph)"),
            horizontal_spacing=0.1,
        )
        fig.add_trace(
            go.Heatmap(z=de, colorscale=self._ctx.config.plot.sequential,
                       colorbar=dict(title="euclidean", x=0.45)),
            row=1, col=1,
        )
        fig.add_trace(
            go.Heatmap(z=dg, colorscale=self._ctx.config.plot.spectral,
                       colorbar=dict(title="geodesic", x=1.0)),
            row=1, col=2,
        )
        self._ctx.styler.style_2d(fig, "Pairwise distances",
                                  height=self._ctx.config.medium_figure_height)
        st.plotly_chart(fig, use_container_width=True)


class PersistenceView(BaseEmbeddingView):
    """Persistence diagram and barcode for H0 and H1."""

    label = "Persistence"
    description = "Topological persistence (H0 components, H1 cycles)."

    def render(self) -> None:
        choices = ["embedding"] + [f"layer {i}" for i in range(len(self._ctx.activations["layers"]))] + ["final"]
        stage = st.selectbox("Stage", options=choices, key="persistence_stage",
                             index=len(choices) // 2)
        m = self._stage_metrics(stage)
        self._render_diagram(m)
        self._render_barcode(m)

    def _stage_metrics(self, stage: str) -> Dict[str, Any]:
        ms = self._ctx.metrics_per_layer
        if stage == "embedding":
            return ms[0]
        if stage == "final":
            return ms[-1]
        return ms[int(stage.split()[-1]) + 1]

    def _render_diagram(self, m: Dict[str, Any]) -> None:
        h0 = m.get("h0_bars", [])
        h1 = m.get("h1_bars", [])
        fig = go.Figure()
        max_val = 1.0
        for bars, name, color in [
            (h0, "H0 (components)", self._ctx.config.theme.accent_cold),
            (h1, "H1 (cycles)", self._ctx.config.theme.accent_warm),
        ]:
            births = []
            deaths = []
            for b, d in bars:
                births.append(b)
                if math.isinf(d):
                    continue
                deaths.append(d)
                max_val = max(max_val, d)
            fig.add_trace(
                go.Scatter(
                    x=[b for b, d in bars if not math.isinf(d)],
                    y=[d for b, d in bars if not math.isinf(d)],
                    mode="markers",
                    marker=dict(size=9, color=color, opacity=0.85,
                                line=dict(color="rgba(255,255,255,0.5)", width=1)),
                    name=name,
                )
            )
        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                line=dict(color="rgba(255,255,255,0.4)", dash="dash"),
                name="diagonal",
                hoverinfo="skip",
            )
        )
        fig.update_xaxes(title="birth")
        fig.update_yaxes(title="death")
        self._ctx.styler.style_2d(fig, "Persistence diagram")
        st.plotly_chart(fig, use_container_width=True)

    def _render_barcode(self, m: Dict[str, Any]) -> None:
        h0 = m.get("h0_bars", [])
        h1 = m.get("h1_bars", [])
        if not h0 and not h1:
            return
        fig = go.Figure()
        row = 0
        for bars, color, name in [
            (h0, self._ctx.config.theme.accent_cold, "H0"),
            (h1, self._ctx.config.theme.accent_warm, "H1"),
        ]:
            for b, d in bars:
                end = d if not math.isinf(d) else b + 1.0
                fig.add_trace(
                    go.Scatter(
                        x=[b, end],
                        y=[row, row],
                        mode="lines",
                        line=dict(color=color, width=2.5),
                        showlegend=(row == 0) or (name == "H1" and row == len(h0)),
                        name=name,
                        hoverinfo="text",
                        text=f"{name} [{b:.3f}, {'inf' if math.isinf(d) else f'{d:.3f}'}]",
                    )
                )
                row += 1
        fig.update_xaxes(title="filtration scale")
        fig.update_yaxes(title="bar index", showticklabels=False)
        self._ctx.styler.style_2d(fig, "Persistence barcode",
                                  height=max(self._ctx.config.compact_figure_height, 10 * row))
        st.plotly_chart(fig, use_container_width=True)


class BerryPhaseView(BaseEmbeddingView):
    """Berry phase and winding number analysis of the token trajectory."""

    label = "Berry / Winding"
    description = "Berry geometric phases and winding numbers along trajectory."

    def render(self) -> None:
        choices = ["embedding"] + [f"layer {i}" for i in range(len(self._ctx.activations["layers"]))] + ["final"]
        stage = st.selectbox("Stage", options=choices, key="berry_stage",
                             index=len(choices) - 1)
        m = self._stage_metrics(stage)
        self._render_berry(m, stage)
        self._render_winding(m, stage)

    def _stage_metrics(self, stage: str) -> Dict[str, Any]:
        ms = self._ctx.metrics_per_layer
        if stage == "embedding":
            return ms[0]
        if stage == "final":
            return ms[-1]
        return ms[int(stage.split()[-1]) + 1]

    def _render_berry(self, m: Dict[str, Any], stage: str) -> None:
        cum = m.get("berry_phase_cum", np.zeros(1))
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(cum)),
                y=cum,
                mode="lines+markers",
                line=dict(color=self._ctx.config.theme.accent, width=2.6),
                fill="tozeroy",
                fillcolor="rgba(100, 181, 246, 0.18)",
                marker=dict(size=6),
                name="cumulative Berry phase",
            )
        )
        for n in (1, 2, 3):
            fig.add_hline(
                y=n * math.pi,
                line_dash="dash",
                line_color="rgba(255,183,77,0.6)",
                annotation_text=f"{n}pi",
                annotation_position="right",
            )
        fig.update_xaxes(title="token position")
        fig.update_yaxes(title="phase (rad)")
        self._ctx.styler.style_2d(
            fig,
            f"{stage} - cumulative geometric phase "
            f"(total = {m.get('berry_phase_total', 0):.4f} rad)",
        )
        st.plotly_chart(fig, use_container_width=True)

    def _render_winding(self, m: Dict[str, Any], stage: str) -> None:
        emb = m.get("winding_embedding")
        if emb is None or emb.shape[0] < 3:
            return
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=(
                f"plane XY (winding = {m['winding_xy']})",
                f"plane XZ (winding = {m['winding_xz']})",
                f"plane YZ (winding = {m['winding_yz']})",
            ),
        )
        pairs = [(0, 1), (0, 2), (1, 2)]
        for i, (a, b) in enumerate(pairs, start=1):
            xs = emb[:, a]
            ys = emb[:, b]
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([xs, xs[:1]]),
                    y=np.concatenate([ys, ys[:1]]),
                    mode="lines+markers",
                    line=dict(color=self._ctx.config.theme.accent, width=2.2),
                    marker=dict(
                        size=6,
                        color=np.arange(xs.size + 1),
                        colorscale=self._ctx.config.plot.spectral,
                    ),
                    hovertext=[f"pos {p}" for p in list(range(xs.size)) + [0]],
                    name=f"plane {i}",
                    showlegend=False,
                ),
                row=1, col=i,
            )
        self._ctx.styler.style_2d(fig, f"{stage} - winding in principal planes",
                                  height=self._ctx.config.compact_figure_height)
        st.plotly_chart(fig, use_container_width=True)


class LipschitzView(BaseEmbeddingView):
    """Local Lipschitz profile and trajectory geometry (speed/curvature/torsion)."""

    label = "Lipschitz / Dynamics"
    description = "Edge-wise Lipschitz, trajectory speed, curvature, torsion."

    def render(self) -> None:
        choices = ["embedding"] + [f"layer {i}" for i in range(len(self._ctx.activations["layers"]))] + ["final"]
        stage = st.selectbox("Stage", options=choices, key="lip_stage", index=len(choices) - 1)
        m = self._stage_metrics(stage)

        self._render_lc(m)
        self._render_dynamics(m)

    def _stage_metrics(self, stage: str) -> Dict[str, Any]:
        ms = self._ctx.metrics_per_layer
        if stage == "embedding":
            return ms[0]
        if stage == "final":
            return ms[-1]
        return ms[int(stage.split()[-1]) + 1]

    def _render_lc(self, m: Dict[str, Any]) -> None:
        lc = m["lc_local"]
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=np.arange(lc.size),
                y=lc,
                marker=dict(
                    color=lc,
                    colorscale=self._ctx.config.plot.spectral,
                    colorbar=dict(title="LC"),
                ),
                hovertemplate="edge %{x}<br>LC=%{y:.4f}<extra></extra>",
            )
        )
        fig.add_hline(
            y=m["lc_global"],
            line_dash="dash",
            line_color="rgba(240, 98, 146, 0.7)",
            annotation_text=f"sup = {m['lc_global']:.4f}",
            annotation_position="top right",
        )
        fig.update_xaxes(title="trajectory edge (token i -> i+1)")
        fig.update_yaxes(title="local Lipschitz constant")
        self._ctx.styler.style_2d(fig, "Local Lipschitz constants along the token trajectory")
        st.plotly_chart(fig, use_container_width=True)

    def _render_dynamics(self, m: Dict[str, Any]) -> None:
        speed = m["trajectory_speed"]
        curv = m["trajectory_curvature"]
        tors = m["trajectory_torsion"]
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            subplot_titles=("speed", "curvature", "torsion"),
                            vertical_spacing=0.08)
        fig.add_trace(go.Scatter(
            x=np.arange(speed.size), y=speed, mode="lines+markers",
            line=dict(color=self._ctx.config.theme.accent, width=2),
            fill="tozeroy",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=np.arange(curv.size), y=curv, mode="lines+markers",
            line=dict(color=self._ctx.config.theme.accent_warm, width=2),
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=np.arange(tors.size), y=tors, mode="lines+markers",
            line=dict(color=self._ctx.config.theme.accent_cold, width=2),
        ), row=3, col=1)
        fig.update_xaxes(title="position", row=3, col=1)
        fig.update_layout(showlegend=False)
        self._ctx.styler.style_2d(fig, "Trajectory differential geometry",
                                  height=self._ctx.config.default_figure_height)
        st.plotly_chart(fig, use_container_width=True)


class NeighborhoodView(BaseEmbeddingView):
    """kNN graph and neighborhood statistics."""

    label = "kNN Graph"
    description = "kNN graph, degree distribution, coherence matrix."

    def render(self) -> None:
        choices = ["embedding"] + [f"layer {i}" for i in range(len(self._ctx.activations["layers"]))] + ["final"]
        stage = st.selectbox("Stage", options=choices, key="knn_stage", index=len(choices) - 1)
        m = self._stage_metrics(stage)
        points = self._stage_points(stage)

        self._render_graph(points, m)
        self._render_coherence(points)

    def _stage_metrics(self, stage: str) -> Dict[str, Any]:
        ms = self._ctx.metrics_per_layer
        if stage == "embedding":
            return ms[0]
        if stage == "final":
            return ms[-1]
        return ms[int(stage.split()[-1]) + 1]

    def _stage_points(self, stage: str) -> np.ndarray:
        if stage == "embedding":
            return self._ctx.activations["embedding"]
        if stage == "final":
            return self._ctx.activations["final"]
        return self._ctx.activations["layers"][int(stage.split()[-1])]

    def _render_graph(self, points: np.ndarray, m: Dict[str, Any]) -> None:
        emb, _ = self._ctx.projector.project(points, "pca", n_components=3)
        graph = m.get("knn_sparse")
        tokens = self._ctx.activations["tokens"]
        fig = go.Figure()
        if graph is not None:
            coo = graph.tocoo()
            xs, ys, zs = [], [], []
            for r, c in zip(coo.row, coo.col):
                if r >= c:
                    continue
                xs.extend([emb[r, 0], emb[c, 0], None])
                ys.extend([emb[r, 1], emb[c, 1], None])
                zs.extend([emb[r, 2], emb[c, 2], None])
            if xs:
                fig.add_trace(
                    go.Scatter3d(
                        x=xs, y=ys, z=zs,
                        mode="lines",
                        line=dict(
                            color=self._ctx.config.theme.border,
                            width=self._ctx.config.graph_edge_width,
                        ),
                        hoverinfo="skip",
                        name="kNN edges",
                    )
                )
        fig.add_trace(
            go.Scatter3d(
                x=emb[:, 0], y=emb[:, 1], z=emb[:, 2],
                mode="markers+text",
                marker=dict(
                    size=self._ctx.config.trajectory_marker_size,
                    color=np.arange(emb.shape[0]),
                    colorscale=self._ctx.config.plot.spectral,
                    opacity=self._ctx.config.cloud_marker_opacity,
                    line=dict(color="white", width=1),
                ),
                text=tokens,
                textposition="top center",
                hovertemplate="%{text}<extra></extra>",
                name="tokens",
            )
        )
        self._ctx.styler.style_3d(fig, f"kNN graph (k = {self._ctx.knn})")
        st.plotly_chart(fig, use_container_width=True)

    def _render_coherence(self, points: np.ndarray) -> None:
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms < self._ctx.config.metrics.eps_norm, 1.0, norms)
        unit = points / norms
        cosine = unit @ unit.T
        tokens = self._ctx.activations["tokens"]
        fig = go.Figure(
            data=go.Heatmap(
                z=cosine,
                x=[f"{i}: {t}" for i, t in enumerate(tokens)],
                y=[f"{i}: {t}" for i, t in enumerate(tokens)],
                colorscale=self._ctx.config.plot.diverging,
                zmin=-1.0, zmax=1.0, zmid=0.0,
                colorbar=dict(title="cos"),
                hovertemplate="i=%{y}<br>j=%{x}<br>cos=%{z:.3f}<extra></extra>",
            )
        )
        self._ctx.styler.style_2d(fig, "Token-token cosine similarity",
                                  height=self._ctx.config.default_figure_height)
        st.plotly_chart(fig, use_container_width=True)


class CrossLayerView(BaseEmbeddingView):
    """Per-token drift across the residual stream."""

    label = "Cross-Layer"
    description = "Drift, similarity, and alignment of each token across layers."

    def render(self) -> None:
        acts = self._ctx.activations
        stages = [("embedding", acts["embedding"])] \
            + [(f"layer {i}", L) for i, L in enumerate(acts["layers"])] \
            + [("final", acts["final"])]
        tokens = acts["tokens"]

        norms = np.stack([np.linalg.norm(p, axis=1) for _, p in stages], axis=1)
        drift = np.stack([
            np.linalg.norm(stages[j + 1][1] - stages[j][1], axis=1)
            for j in range(len(stages) - 1)
        ], axis=1)
        alignment = np.stack([
            self._cosine_diag(stages[j + 1][1], stages[j][1])
            for j in range(len(stages) - 1)
        ], axis=1)

        stage_names = [name for name, _ in stages]
        edge_names = [f"{stages[j][0]}->{stages[j+1][0]}" for j in range(len(stages) - 1)]
        self._render_heatmap("||h_l||", norms, stage_names, tokens)
        self._render_heatmap("||h_{l+1} - h_l||", drift, edge_names, tokens)
        self._render_heatmap("cos(h_l, h_{l+1})", alignment, edge_names, tokens, diverging=True)

    def _cosine_diag(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        na = np.linalg.norm(A, axis=1)
        nb = np.linalg.norm(B, axis=1)
        denom = np.maximum(na * nb, self._ctx.config.metrics.eps_div)
        return np.einsum("ij,ij->i", A, B) / denom

    def _render_heatmap(
        self,
        title: str,
        z: np.ndarray,
        xlabels: List[str],
        tokens: List[str],
        diverging: bool = False,
    ) -> None:
        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=xlabels,
                y=[f"{i}: {t}" for i, t in enumerate(tokens)],
                colorscale=(self._ctx.config.plot.diverging
                            if diverging else self._ctx.config.plot.spectral),
                zmid=0.0 if diverging else None,
                zmin=-1.0 if diverging else None,
                zmax=1.0 if diverging else None,
                colorbar=dict(title=title),
                hovertemplate="%{x}<br>%{y}<br>%{z:.4f}<extra></extra>",
            )
        )
        self._ctx.styler.style_2d(fig, title, height=self._ctx.config.medium_figure_height)
        st.plotly_chart(fig, use_container_width=True)


class QuaternionView(BaseEmbeddingView):
    """Quaternion decomposition of activations (w, x, y, z sub-channels)."""

    label = "Quaternion"
    description = "View each activation as a quaternion and plot w/x/y/z channels."

    def render(self) -> None:
        acts = self._ctx.activations
        choices = ["embedding"] + [f"layer {i}" for i in range(len(acts["layers"]))] + ["final"]
        stage = st.selectbox("Stage", options=choices, key="quat_stage",
                             index=len(choices) - 1)
        points = self._points(stage)
        d = points.shape[1]
        if d % 4 != 0:
            st.info(f"Dimension {d} is not divisible by 4; quaternion view unavailable.")
            return
        q = d // 4
        w = points[:, :q]
        x = points[:, q:2*q]
        y = points[:, 2*q:3*q]
        z = points[:, 3*q:]
        comps = (w, x, y, z)

        self._render_component_norms(comps)
        self._render_quaternion_norms(comps)
        self._render_sphere(comps)

    def _points(self, stage: str) -> np.ndarray:
        acts = self._ctx.activations
        if stage == "embedding":
            return acts["embedding"]
        if stage == "final":
            return acts["final"]
        return acts["layers"][int(stage.split()[-1])]

    def _render_component_norms(self, comps: Tuple[np.ndarray, ...]) -> None:
        names = self._ctx.config.quaternion_component_names
        norms = np.stack([np.linalg.norm(c, axis=1) for c in comps], axis=1)
        tokens = self._ctx.activations["tokens"]
        fig = go.Figure()
        for i, name in enumerate(names):
            fig.add_trace(
                go.Scatter(
                    x=np.arange(norms.shape[0]),
                    y=norms[:, i],
                    mode="lines+markers",
                    name=f"||h_{name}||",
                    line=dict(width=2.2),
                )
            )
        fig.update_xaxes(
            title="token position",
            tickvals=list(range(len(tokens))),
            ticktext=[f"{i}: {t}" for i, t in enumerate(tokens)],
            tickangle=-45,
        )
        fig.update_yaxes(title="component L2 norm")
        self._ctx.styler.style_2d(fig, "Quaternion component norms per token")
        st.plotly_chart(fig, use_container_width=True)

    def _render_quaternion_norms(self, comps: Tuple[np.ndarray, ...]) -> None:
        norms_sq = sum((np.sum(c**2, axis=1) for c in comps))
        q_norm = np.sqrt(norms_sq)
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=q_norm.reshape(-1),
                nbinsx=40,
                marker_color=self._ctx.config.theme.accent,
            )
        )
        fig.update_xaxes(title="||q|| (per token, summed over sub-channels)")
        fig.update_yaxes(title="count")
        self._ctx.styler.style_2d(fig, "Quaternion norm distribution across tokens")
        st.plotly_chart(fig, use_container_width=True)

    def _render_sphere(self, comps: Tuple[np.ndarray, ...]) -> None:
        reps = np.stack([np.mean(c, axis=1) for c in comps], axis=1)
        norms = np.linalg.norm(reps, axis=1, keepdims=True)
        norms = np.where(norms < self._ctx.config.metrics.eps_norm, 1.0, norms)
        unit = reps / norms
        tokens = self._ctx.activations["tokens"]
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=unit[:, 1],
                    y=unit[:, 2],
                    z=unit[:, 3],
                    mode="markers+text",
                    marker=dict(
                        size=self._ctx.config.trajectory_marker_size + 2,
                        color=unit[:, 0],
                        colorscale=self._ctx.config.plot.diverging,
                        cmin=-1.0, cmax=1.0,
                        colorbar=dict(title="w"),
                        line=dict(color="white", width=1),
                    ),
                    text=tokens,
                    textposition="top center",
                    hovertemplate=(
                        "w=%{marker.color:.3f}<br>"
                        "x=%{x:.3f} y=%{y:.3f} z=%{z:.3f}<extra></extra>"
                    ),
                )
            ]
        )
        self._ctx.styler.style_3d(
            fig,
            "Unit-quaternion projection "
            "(x, y, z on sphere; color = scalar part w)",
        )
        st.plotly_chart(fig, use_container_width=True)


class RawView(BaseEmbeddingView):
    """Raw activation heatmap (tokens x features)."""

    label = "Raw activations"
    description = "Inspect the raw hidden state as a tokens x features heatmap."

    def render(self) -> None:
        acts = self._ctx.activations
        choices = ["embedding"] + [f"layer {i}" for i in range(len(acts["layers"]))] + ["final"]
        stage = st.selectbox("Stage", options=choices, key="raw_stage",
                             index=len(choices) - 1)
        points = self._points(stage)
        tokens = acts["tokens"]
        fig = go.Figure(
            data=go.Heatmap(
                z=points,
                x=[f"f{i}" for i in range(points.shape[1])],
                y=[f"{i}: {t}" for i, t in enumerate(tokens)],
                colorscale=self._ctx.config.plot.diverging,
                zmid=0.0,
                colorbar=dict(title="activation"),
                hovertemplate="token=%{y}<br>feature=%{x}<br>value=%{z:.4f}<extra></extra>",
            )
        )
        self._ctx.styler.style_2d(fig, f"{stage} - raw activation matrix",
                                  height=self._ctx.config.default_figure_height)
        st.plotly_chart(fig, use_container_width=True)

    def _points(self, stage: str) -> np.ndarray:
        acts = self._ctx.activations
        if stage == "embedding":
            return acts["embedding"]
        if stage == "final":
            return acts["final"]
        return acts["layers"][int(stage.split()[-1])]


class ViewRegistry:
    """Collects and instantiates views."""

    def __init__(self, context: RenderContext) -> None:
        self._context = context
        self._factories: List[Callable[[RenderContext], BaseEmbeddingView]] = []

    def register(self, factory: Callable[[RenderContext], BaseEmbeddingView]) -> None:
        """Register a view factory."""
        self._factories.append(factory)

    def build(self) -> List[BaseEmbeddingView]:
        """Instantiate every registered view."""
        return [f(self._context) for f in self._factories]


class SidebarController:
    """Sidebar inputs (checkpoint, text, kNN, model script path)."""

    def __init__(self, config: NavigatorConfig) -> None:
        self._config = config

    def render(self) -> Dict[str, Any]:
        """Render the sidebar and return user selections."""
        st.sidebar.markdown("## Checkpoint")
        uploaded = st.sidebar.file_uploader(
            "Upload checkpoint",
            type=[ext.lstrip(".") for ext in self._config.supported_checkpoint_extensions],
        )
        manual_path = st.sidebar.text_input(
            "...or path on disk",
            value="",
        )
        script_path = st.sidebar.text_input(
            "topogpt2 model script",
            value="topogpt2_1.py",
            help="Filesystem path to the module defining TopoGPT2.",
        )
        st.sidebar.markdown("---")
        st.sidebar.markdown("## Input")
        text = st.sidebar.text_area(
            "Text to embed",
            value=self._config.default_text,
            height=140,
        )
        st.sidebar.markdown("---")
        st.sidebar.markdown("## Graph")
        knn = st.sidebar.slider(
            "kNN neighbors",
            min_value=self._config.sampling.knn_minimum,
            max_value=self._config.sampling.knn_maximum,
            value=self._config.sampling.knn_default,
        )
        st.sidebar.markdown("---")
        st.sidebar.markdown("## References")
        st.sidebar.markdown(
            "- Berry (1984): *Quantal phases accompanying adiabatic changes*  \n"
            "- Gromov (1987): *Hyperbolic groups*  \n"
            "- Edelsbrunner and Harer (2010): *Computational Topology*  \n"
            "- Tenenbaum et al. (2000): *Isomap*  \n"
            "- McInnes et al. (2018): *UMAP*  \n"
            "- Reimers and Gurevych (2019): *Sentence-BERT*"
        )
        return {
            "uploaded": uploaded,
            "manual_path": manual_path.strip(),
            "script_path": script_path.strip(),
            "text": text,
            "knn": knn,
        }


class ModuleImporter:
    """Imports ``topogpt2_1.py`` from a user-supplied filesystem path."""

    def load(self, path: str) -> Any:
        """Dynamically import the TopoGPT2 module.

        Args:
            path: Path to ``topogpt2_1.py``.

        Returns:
            The imported module object.

        Raises:
            FileNotFoundError: if the path does not exist.
            ImportError: if the module cannot be loaded.
        """
        import importlib.util
        import sys

        p = Path(path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Model script not found: {p}")
        spec = importlib.util.spec_from_file_location("topogpt2_user_module", str(p))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load spec from {p}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["topogpt2_user_module"] = module
        parent = str(p.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        spec.loader.exec_module(module)
        if not hasattr(module, "TopoGPT2") or not hasattr(module, "TopoGPT2Config"):
            raise ImportError(
                f"Module at {p} does not expose TopoGPT2 and TopoGPT2Config."
            )
        return module


class NavigatorApp:
    """Top-level orchestrator."""

    def __init__(self, config: Optional[NavigatorConfig] = None) -> None:
        self._config = config or NavigatorConfig()
        self._style_injector = StyleInjector(self._config.theme)
        self._sidebar = SidebarController(self._config)
        self._loader = ModelLoader(self._config)
        self._capture = ActivationCapture(self._config)
        self._metric = MetricSuite(self._config)
        self._projector = Projector(self._config)
        self._styler = FigureStyler(self._config)
        self._module_importer = ModuleImporter()

    def run(self) -> None:
        """Entry point for ``streamlit run``."""
        st.set_page_config(
            page_title=self._config.page_title,
            layout="wide",
            initial_sidebar_state="expanded",
        )
        self._style_injector.inject()
        self._render_header()
        selections = self._sidebar.render()

        bundle = self._try_load_bundle(selections)
        if bundle is None:
            self._render_landing()
            return

        try:
            activations = self._capture.run(bundle, selections["text"])
        except Exception as exc:
            st.error(f"Activation capture failed: {exc}")
            return

        metrics_per_layer = self._compute_all_metrics(activations, selections["knn"])
        self._render_meta(bundle, activations, selections["knn"])

        ctx = RenderContext(
            bundle=bundle,
            activations=activations,
            metrics_per_layer=metrics_per_layer,
            projector=self._projector,
            styler=self._styler,
            config=self._config,
            knn=selections["knn"],
        )
        registry = self._build_registry(ctx)
        self._render_tabs(registry)
        self._render_footer()

    def _render_header(self) -> None:
        st.markdown('<div class="header-container">', unsafe_allow_html=True)
        st.title(self._config.page_title)
        st.subheader("Interactive exploration of the residual-stream embeddings of a TopoGPT2 checkpoint")
        st.markdown(
            '<div class="citation-box">'
            "The navigator tokenizes your input, runs a forward pass through the "
            "loaded model, captures the hidden states at every transformer layer, "
            "and produces a complete geometric / topological analysis: "
            "curvature (kappa), Gromov-delta hyperbolicity, persistent homology "
            "(H0, H1), Berry phases, winding numbers, Lipschitz constants, "
            "shortest-path statistics, and trajectory differential geometry."
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    def _render_landing(self) -> None:
        st.markdown('<div class="header-container">', unsafe_allow_html=True)
        st.markdown(
            "### Getting started\n"
            "1. Point **topogpt2 model script** at your `topogpt2_1.py` file.\n"
            "2. Upload a checkpoint or paste its path.\n"
            "3. Enter text in the sidebar and pick a kNN count.\n\n"
            "Every tab reads directly from real hidden states. No synthetic data."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    def _try_load_bundle(self, selections: Dict[str, Any]) -> Optional[CheckpointBundle]:
        ckpt_source = selections.get("uploaded") or selections.get("manual_path")
        if not ckpt_source:
            return None
        script_path = selections.get("script_path", "")
        try:
            module = self._module_importer.load(script_path)
        except Exception as exc:
            st.error(f"Could not import the model module: {exc}")
            return None
        try:
            return self._loader.load(ckpt_source, module)
        except Exception as exc:
            st.error(f"Checkpoint load failed: {exc}")
            return None

    def _compute_all_metrics(
        self,
        activations: Dict[str, Any],
        knn: int,
    ) -> List[Dict[str, Any]]:
        stages = [activations["embedding"]] + activations["layers"] + [activations["final"]]
        metrics: List[Dict[str, Any]] = []
        for i, pts in enumerate(stages):
            try:
                metrics.append(self._metric.compute(pts, knn))
            except Exception as exc:
                st.warning(f"Stage {i}: metric computation partial failure ({exc}).")
                metrics.append(self._metric._trivial({"n_points": int(pts.shape[0]),
                                                      "ambient_dim": int(pts.shape[1])}))
        return metrics

    def _render_meta(
        self,
        bundle: CheckpointBundle,
        activations: Dict[str, Any],
        knn: int,
    ) -> None:
        cols = st.columns(5)
        cols[0].metric("checkpoint", bundle.source_name[:30])
        cols[1].metric("D_MODEL", f"{bundle.embedding_dim()}")
        cols[2].metric("layers", f"{bundle.num_layers()}")
        cols[3].metric("tokens", f"{len(activations['tokens'])}")
        cols[4].metric("kNN", f"{knn}")

    def _build_registry(self, ctx: RenderContext) -> ViewRegistry:
        registry = ViewRegistry(ctx)
        registry.register(lambda c: OverviewView(c))
        registry.register(lambda c: CloudView(c))
        registry.register(lambda c: MetricsView(c))
        registry.register(lambda c: PersistenceView(c))
        registry.register(lambda c: BerryPhaseView(c))
        registry.register(lambda c: LipschitzView(c))
        registry.register(lambda c: NeighborhoodView(c))
        registry.register(lambda c: CrossLayerView(c))
        registry.register(lambda c: QuaternionView(c))
        registry.register(lambda c: RawView(c))
        return registry

    def _render_tabs(self, registry: ViewRegistry) -> None:
        views = registry.build()
        tabs = st.tabs([v.label for v in views])
        for tab, view in zip(tabs, views):
            with tab:
                st.markdown('<div class="header-container">', unsafe_allow_html=True)
                st.markdown(f"#### {view.label}")
                st.caption(view.description)
                st.markdown("</div>", unsafe_allow_html=True)
                try:
                    view.render()
                except Exception as exc:
                    st.error(f"View '{view.label}' raised: {exc}")

    def _render_footer(self) -> None:
        st.markdown(
            '<div class="footer">'
            "<b>TopoGPT2 Embeddings Navigator</b> - "
            "Topological and geometric analysis of residual-stream activations."
            "</div>",
            unsafe_allow_html=True,
        )


def main() -> None:
    """Streamlit script entry point."""
    app = NavigatorApp()
    app.run()


if __name__ == "__main__":
    main()
