"""TopoGPT2 Weights Explorer.

Production-grade interactive 3D/2D visualization system for inspecting the
learned parameter tensors of a TopoGPT2 checkpoint.

The explorer is a single Streamlit application that loads a `safetensors` or
`.pt` checkpoint produced by `topogpt2_1.py` and surfaces every meaningful
view of the model's parameter space:

  * Global parameter inventory (type, shape, norm, sparsity).
  * Per-tensor 3D point clouds (PCA, t-SNE-approximating random projection).
  * Per-tensor 2D heatmaps and correlation matrices.
  * Quaternion weight decomposition (w, x, y, z component analysis).
  * Spectral kernel visualization in the complex frequency plane.
  * Torus topology graph with learned edge quaternions and node embeddings.
  * Attention projection analysis (Q/K/V/O per head).
  * MoE router logit landscape and expert utilization proxies.
  * Global mechanistic metrics (participation ratio, entropy, spectral
    flatness, effective rank, fractal dimension, coherence).

The application follows SOLID:

  * Single responsibility - one class per concern (loading, metrics,
    visualization, layout, styling, configuration).
  * Open/closed - new visualizers subclass `BaseVisualizer` and register via
    the `VisualizerRegistry` without touching existing code.
  * Liskov - every visualizer honours the same render contract.
  * Interface segregation - small focused protocols per concern.
  * Dependency inversion - the page composes abstractions, never concrete
    implementations directly.

All numeric thresholds, sample caps, and rendering parameters live in
`ExplorerConfig`. No magic numbers appear inline.
"""

from __future__ import annotations

import io
import json
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
from plotly.subplots import make_subplots
from scipy import fft as scipy_fft
from scipy import signal as scipy_signal
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection


@dataclass(frozen=True)
class ThemeTokens:
    """Design tokens for the explorer's dark scientific theme."""

    background_gradient_start: str = "#0a0e17"
    background_gradient_end: str = "#0d1b2a"
    panel_background: str = "rgba(15, 32, 61, 0.85)"
    card_background: str = "rgba(26, 42, 85, 0.70)"
    accent_primary: str = "#64b5f6"
    accent_secondary: str = "#4a86e8"
    border_primary: str = "#4a6fa5"
    border_secondary: str = "#3a5ba0"
    text_primary: str = "#e0e0ff"
    text_muted: str = "#8a9bbd"
    glow_shadow: str = "0 0 10px rgba(100, 181, 246, 0.35)"


@dataclass(frozen=True)
class PlotTheme:
    """Plot-level theme constants."""

    template: str = "plotly_dark"
    sequential_scale: str = "Viridis"
    diverging_scale: str = "RdBu"
    spectral_scale: str = "Plasma"
    font_color: str = "#e0e0ff"
    grid_color: str = "#4a6fa5"
    paper_bgcolor: str = "rgba(0, 0, 0, 0)"
    plot_bgcolor: str = "rgba(10, 14, 23, 0.3)"
    marker_line_color: str = "rgba(100, 181, 246, 0.35)"


@dataclass(frozen=True)
class SamplingLimits:
    """Hard caps to keep the UI responsive regardless of model size."""

    max_rows_for_pca: int = 4096
    max_cols_for_pca: int = 4096
    max_elements_for_histogram: int = 500_000
    max_rows_for_correlation: int = 256
    max_elements_heatmap: int = 256 * 256
    max_fft_rows: int = 64
    max_fft_samples_per_row: int = 4096
    min_pca_components: int = 3
    default_fractal_variance_floor: float = 1e-3
    global_scan_max_rows: int = 512
    global_scan_max_cols: int = 512


@dataclass(frozen=True)
class MetricsConfig:
    """Numerical stability thresholds for metrics."""

    histogram_bins: int = 60
    entropy_epsilon: float = 1e-10
    norm_epsilon: float = 1e-12
    spectral_flatness_epsilon: float = 1e-10
    effective_rank_cutoff: float = 1e-6
    coherence_diagonal_value: float = 0.0


@dataclass(frozen=True)
class GenerationLimits:
    """Random-projection and synthetic probe parameters."""

    random_projection_eps: float = 0.25
    random_state: int = 42
    routing_probe_tokens: int = 512


@dataclass(frozen=True)
class ExplorerConfig:
    """Top-level configuration container."""

    theme: ThemeTokens = field(default_factory=ThemeTokens)
    plot: PlotTheme = field(default_factory=PlotTheme)
    sampling: SamplingLimits = field(default_factory=SamplingLimits)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    generation: GenerationLimits = field(default_factory=GenerationLimits)

    page_title: str = "TopoGPT2 Weights Explorer"
    page_icon: str = ":microscope:"
    default_figure_height: int = 620
    compact_figure_height: int = 340
    medium_figure_height: int = 460
    point_cloud_marker_size: int = 3
    point_cloud_opacity: float = 0.72
    graph_node_size: int = 22
    graph_edge_width: float = 2.4
    scatter_hover_digits: int = 4

    torus_radial_bins_default: int = 2
    torus_angular_bins_default: int = 4

    quaternion_component_names: Tuple[str, str, str, str] = ("w", "x", "y", "z")
    quaternion_component_colors: Tuple[str, str, str, str] = (
        "#64b5f6",
        "#f06292",
        "#81c784",
        "#ffb74d",
    )

    supported_checkpoint_extensions: Tuple[str, ...] = (
        ".safetensors",
        ".pt",
        ".bin",
        ".pth",
    )


class StyleInjector:
    """Injects the global CSS for the explorer into the Streamlit page."""

    def __init__(self, theme: ThemeTokens) -> None:
        self._theme = theme

    def _build_css(self) -> str:
        t = self._theme
        return f"""
        <style>
        .main {{
            background: linear-gradient(135deg, {t.background_gradient_start} 0%, {t.background_gradient_end} 100%);
            color: {t.text_primary};
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}
        .header-container {{
            background: {t.panel_background};
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            border: 1px solid {t.border_primary};
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        }}
        .metric-card {{
            background: {t.card_background};
            border-radius: 12px;
            padding: 1rem;
            border: 1px solid {t.border_secondary};
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        .metric-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 15px rgba(58, 91, 160, 0.35);
        }}
        .scientific-notation {{
            font-family: 'Lucida Console', Monaco, monospace;
            background: rgba(30, 45, 80, 0.6);
            padding: 0.4rem 0.7rem;
            border-radius: 6px;
            border-left: 3px solid {t.accent_secondary};
        }}
        .citation-box {{
            background: rgba(22, 38, 68, 0.8);
            border-left: 4px solid {t.accent_primary};
            padding: 1rem;
            margin: 1rem 0;
            font-style: italic;
            color: #bbdefb;
        }}
        .theory-box {{
            background: rgba(19, 41, 77, 0.85);
            border: 1px solid #5c9bd5;
            border-radius: 10px;
            padding: 1.2rem;
            margin: 1rem 0;
        }}
        h1, h2, h3 {{
            color: {t.accent_primary} !important;
            text-shadow: {t.glow_shadow};
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

    def inject(self) -> None:
        """Render the CSS block inside the current Streamlit page."""
        st.markdown(self._build_css(), unsafe_allow_html=True)


class TensorClassifier:
    """Classifies tensor keys from a TopoGPT2 checkpoint by semantic role.

    The classifier parses the parameter name and extracts:
      * a coarse role ("attention", "moe_router", "spectral_kernel", etc.)
      * the layer index if present
      * the quaternion component (w, x, y, z) if present
      * whether the tensor represents a frequency-domain kernel
    """

    _LAYER_PATTERN = re.compile(r"layers\.(\d+)")
    _QUATERNION_COMPONENT_PATTERN = re.compile(r"(^|[._])([wxyz])(_|$)")
    _SPECTRAL_KERNEL_PATTERN = re.compile(r"\b(kr|ki|enc_kr|enc_ki|dec_kr|dec_ki)\b")

    def classify(self, name: str, shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Return structured metadata for a checkpoint tensor.

        Args:
            name: Full dotted parameter name.
            shape: Tensor shape.

        Returns:
            A dictionary with keys ``role``, ``layer``, ``component``,
            ``is_spectral``, ``is_complex_kernel``, ``shape``.
        """
        role = self._classify_role(name)
        layer = self._extract_layer(name)
        component = self._extract_quaternion_component(name)
        is_complex_kernel = bool(self._SPECTRAL_KERNEL_PATTERN.search(name))
        is_spectral = is_complex_kernel or "spectral" in name.lower()
        return {
            "name": name,
            "role": role,
            "layer": layer,
            "component": component,
            "is_spectral": is_spectral,
            "is_complex_kernel": is_complex_kernel,
            "shape": tuple(shape),
            "ndim": len(shape),
            "numel": int(np.prod(shape)) if len(shape) > 0 else 1,
        }

    def _classify_role(self, name: str) -> str:
        lowered = name.lower()
        if "token_embed" in lowered:
            return "token_embedding"
        if "lm_head" in lowered:
            return "lm_head"
        if "final_norm" in lowered:
            return "final_norm"
        if "node_embed" in lowered:
            return "torus_node_embedding"
        if "edge_quat" in lowered:
            return "torus_edge_quaternion"
        if "router" in lowered:
            return "moe_router"
        if "experts." in lowered:
            return "moe_expert"
        if "shared_expert" in lowered and "spectral_ae" in lowered:
            return "spectral_autoencoder"
        if "torus_spectral" in lowered:
            return "torus_spectral"
        if "rope" in lowered or "rotary" in lowered:
            return "rope"
        if "q_proj" in lowered:
            return "attn_q"
        if "k_proj" in lowered:
            return "attn_k"
        if "v_proj" in lowered:
            return "attn_v"
        if "o_proj" in lowered:
            return "attn_o"
        if "norm" in lowered:
            return "norm"
        if "temperature" in lowered:
            return "attn_temperature"
        if "readout" in lowered:
            return "readout"
        if "torus_proj" in lowered:
            return "torus_projection"
        if any(tag in lowered for tag in ("ww.", "wx.", "wy.", "wz.")):
            return "quaternion_linear"
        return "other"

    def _extract_layer(self, name: str) -> Optional[int]:
        m = self._LAYER_PATTERN.search(name)
        return int(m.group(1)) if m else None

    def _extract_quaternion_component(self, name: str) -> Optional[str]:
        m = self._QUATERNION_COMPONENT_PATTERN.search(name)
        return m.group(2) if m else None


class CheckpointLoader:
    """Loads raw tensor dictionaries from disk.

    Supports both ``safetensors`` files and plain PyTorch pickles produced
    by ``torch.save``. The loader returns CPU float32 tensors exclusively to
    guarantee downstream compatibility with NumPy.
    """

    def __init__(self, config: ExplorerConfig) -> None:
        self._config = config

    def load(self, source: Any) -> Dict[str, torch.Tensor]:
        """Load a checkpoint from a file path or an uploaded file-like.

        Args:
            source: Either a path (``str`` or ``Path``) or a Streamlit
                ``UploadedFile`` object.

        Returns:
            Dictionary mapping tensor name to CPU ``torch.Tensor``.

        Raises:
            ValueError: If the extension is not supported.
            RuntimeError: If deserialization fails.
        """
        filename, buffer = self._materialize(source)
        suffix = Path(filename).suffix.lower()
        if suffix not in self._config.supported_checkpoint_extensions:
            raise ValueError(
                f"Unsupported extension '{suffix}'. "
                f"Allowed: {self._config.supported_checkpoint_extensions}."
            )
        if suffix == ".safetensors":
            return self._load_safetensors(buffer)
        return self._load_torch(buffer)

    def _materialize(self, source: Any) -> Tuple[str, io.BytesIO]:
        if hasattr(source, "read") and hasattr(source, "name"):
            data = source.read()
            return source.name, io.BytesIO(data)
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path.name, io.BytesIO(path.read_bytes())

    def _load_safetensors(self, buffer: io.BytesIO) -> Dict[str, torch.Tensor]:
        from safetensors.torch import load as safetensors_load

        buffer.seek(0)
        raw = safetensors_load(buffer.read())
        return {k: self._to_cpu_float32(v) for k, v in raw.items()}

    def _load_torch(self, buffer: io.BytesIO) -> Dict[str, torch.Tensor]:
        buffer.seek(0)
        try:
            obj = torch.load(buffer, map_location="cpu", weights_only=True)
        except Exception:
            buffer.seek(0)
            obj = torch.load(buffer, map_location="cpu", weights_only=False)
        state_dict = self._extract_state_dict(obj)
        return {k: self._to_cpu_float32(v) for k, v in state_dict.items()}

    def _extract_state_dict(self, obj: Any) -> Dict[str, torch.Tensor]:
        if isinstance(obj, dict):
            for key in ("model_state_dict", "state_dict", "model", "weights"):
                if key in obj and isinstance(obj[key], dict):
                    candidate = obj[key]
                    if self._looks_like_state_dict(candidate):
                        return candidate
            if self._looks_like_state_dict(obj):
                return obj
        raise RuntimeError(
            "Could not locate a tensor state dictionary inside the checkpoint."
        )

    @staticmethod
    def _looks_like_state_dict(obj: Any) -> bool:
        if not isinstance(obj, dict) or not obj:
            return False
        sample = next(iter(obj.values()))
        return isinstance(sample, torch.Tensor)

    @staticmethod
    def _to_cpu_float32(tensor: torch.Tensor) -> torch.Tensor:
        t = tensor.detach().cpu()
        if t.is_complex():
            return t
        if t.dtype != torch.float32:
            t = t.to(torch.float32)
        return t


class TensorInventory:
    """Holds a checkpoint's tensors along with per-tensor metadata."""

    def __init__(
        self,
        tensors: Dict[str, torch.Tensor],
        classifier: TensorClassifier,
    ) -> None:
        self._tensors = tensors
        self._metadata: Dict[str, Dict[str, Any]] = {
            name: classifier.classify(name, tuple(t.shape))
            for name, t in tensors.items()
        }

    def names(self) -> List[str]:
        """Return all tensor names sorted alphabetically."""
        return sorted(self._tensors.keys())

    def tensor(self, name: str) -> torch.Tensor:
        """Retrieve a tensor by name."""
        return self._tensors[name]

    def meta(self, name: str) -> Dict[str, Any]:
        """Retrieve structured metadata for a tensor."""
        return self._metadata[name]

    def layers(self) -> List[int]:
        """Return all unique layer indices present in the checkpoint."""
        values = {m["layer"] for m in self._metadata.values() if m["layer"] is not None}
        return sorted(values)

    def roles(self) -> List[str]:
        """Return the distinct roles present in the checkpoint."""
        return sorted({m["role"] for m in self._metadata.values()})

    def filter(
        self,
        role: Optional[str] = None,
        layer: Optional[int] = None,
        component: Optional[str] = None,
        spectral_only: bool = False,
    ) -> List[str]:
        """Return tensor names matching the supplied filters."""
        out: List[str] = []
        for name, m in self._metadata.items():
            if role is not None and m["role"] != role:
                continue
            if layer is not None and m["layer"] != layer:
                continue
            if component is not None and m["component"] != component:
                continue
            if spectral_only and not m["is_spectral"]:
                continue
            out.append(name)
        return sorted(out)

    def total_parameters(self) -> int:
        """Total parameter count across the checkpoint."""
        return int(sum(m["numel"] for m in self._metadata.values()))

    def summary_rows(self) -> List[Dict[str, Any]]:
        """Return a list of per-tensor rows suitable for a Streamlit table."""
        rows: List[Dict[str, Any]] = []
        for name in self.names():
            m = self._metadata[name]
            t = self._tensors[name]
            arr = t.float().numpy() if t.is_complex() else t.numpy()
            rows.append(
                {
                    "name": name,
                    "role": m["role"],
                    "layer": m["layer"] if m["layer"] is not None else "-",
                    "component": m["component"] if m["component"] else "-",
                    "shape": str(m["shape"]),
                    "ndim": m["ndim"],
                    "numel": m["numel"],
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "abs_max": float(np.max(np.abs(arr))),
                    "is_complex": bool(t.is_complex()),
                    "is_spectral": m["is_spectral"],
                }
            )
        return rows


class TensorProjector:
    """Projects arbitrary tensors into 2D matrices and 3D point clouds.

    Responsibilities:
      * Reduce N-dimensional tensors to a 2D matrix of rows (samples) vs
        columns (features) while preserving interpretability.
      * Subsample rows/columns to respect the configured limits.
      * Compute 3D embeddings (PCA or Gaussian random projection) with
        deterministic seeding.
    """

    def __init__(self, config: ExplorerConfig) -> None:
        self._config = config
        self._sampling = config.sampling
        self._generation = config.generation

    def to_matrix(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert a tensor to a 2D ``float64`` matrix.

        Complex tensors are stacked as ``[real, imag]`` along the feature axis
        before flattening, which preserves their dimensionality information.
        """
        t = tensor.detach()
        if t.is_complex():
            stacked = torch.stack([t.real, t.imag], dim=-1)
            arr = stacked.cpu().numpy()
        else:
            arr = t.cpu().numpy()
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim > 2:
            leading = arr.shape[0]
            trailing = int(np.prod(arr.shape[1:]))
            arr = arr.reshape(leading, trailing)
        return np.asarray(arr, dtype=np.float64)

    def subsample(
        self,
        matrix: np.ndarray,
        max_rows: Optional[int] = None,
        max_cols: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Return a row/column subsample bounded by the configured caps."""
        rng = np.random.default_rng(
            seed if seed is not None else self._generation.random_state
        )
        rows_cap = max_rows or self._sampling.max_rows_for_pca
        cols_cap = max_cols or self._sampling.max_cols_for_pca
        n_rows, n_cols = matrix.shape
        if n_rows > rows_cap:
            idx = rng.choice(n_rows, size=rows_cap, replace=False)
            matrix = matrix[idx]
        if n_cols > cols_cap:
            idx = rng.choice(n_cols, size=cols_cap, replace=False)
            matrix = matrix[:, idx]
        return matrix

    def project_3d(
        self,
        matrix: np.ndarray,
        method: str = "pca",
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Project a matrix to 3D using PCA or Gaussian random projection.

        Args:
            matrix: Row-major 2D matrix ``[N, F]``.
            method: Either ``"pca"`` or ``"random"``.

        Returns:
            A pair ``(embedding[N, 3], info)`` where ``info`` contains the
            explained variance ratio (PCA) or the Johnson-Lindenstrauss
            ``eps`` used (random projection).
        """
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        n_components = max(self._sampling.min_pca_components, 3)
        if matrix.shape[0] < n_components or matrix.shape[1] < n_components:
            pad_rows = max(0, n_components - matrix.shape[0])
            pad_cols = max(0, n_components - matrix.shape[1])
            if pad_rows > 0:
                matrix = np.vstack([matrix, np.zeros((pad_rows, matrix.shape[1]))])
            if pad_cols > 0:
                matrix = np.hstack([matrix, np.zeros((matrix.shape[0], pad_cols))])
        if method == "pca":
            return self._pca_3d(matrix)
        if method == "random":
            return self._random_3d(matrix)
        raise ValueError(f"Unknown projection method: {method}")

    def _pca_3d(self, matrix: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        model = PCA(n_components=3, random_state=self._generation.random_state)
        emb = model.fit_transform(matrix)
        info = {
            "explained_variance_ratio": float(np.sum(model.explained_variance_ratio_)),
            "pc1": float(model.explained_variance_ratio_[0]),
            "pc2": float(model.explained_variance_ratio_[1]),
            "pc3": float(model.explained_variance_ratio_[2]),
        }
        return emb, info

    def _random_3d(self, matrix: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        model = GaussianRandomProjection(
            n_components=3,
            eps=self._generation.random_projection_eps,
            random_state=self._generation.random_state,
        )
        try:
            emb = model.fit_transform(matrix)
        except ValueError:
            model = GaussianRandomProjection(
                n_components=3,
                random_state=self._generation.random_state,
            )
            emb = model.fit_transform(matrix)
        info = {"eps": float(self._generation.random_projection_eps)}
        return emb, info


class MetricCalculator:
    """Computes mechanistic-interpretability metrics for parameter matrices.

    Results are memoized by the ``id`` of the input array to guarantee that
    repeated requests for the same subsampled matrix (common across tabs) do
    not re-run expensive SVDs.
    """

    def __init__(self, config: ExplorerConfig) -> None:
        self._config = config
        self._metrics_cfg = config.metrics
        self._sampling = config.sampling
        self._cache: Dict[Tuple[int, Tuple[int, ...]], Dict[str, float]] = {}

    def compute_all(self, matrix: np.ndarray) -> Dict[str, float]:
        """Compute the full metrics dictionary in one pass.

        Uses a lightweight cache keyed by ``(id(matrix), shape)`` so that
        recomputation across visualizers is avoided when the same subsampled
        matrix is analyzed multiple times in one render cycle.
        """
        cache_key = (id(matrix), matrix.shape)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        clean = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        svd_values = self._svd_safe(clean)
        coherence_max, coherence_mean = self.coherence(clean)
        result = {
            "mean": float(np.mean(clean)),
            "std": float(np.std(clean)),
            "abs_max": float(np.max(np.abs(clean))),
            "frobenius_norm": float(np.linalg.norm(clean)),
            "sparsity": self.sparsity(clean),
            "entropy": self.entropy(clean),
            "effective_rank": self._effective_rank_from_svd(svd_values, clean.shape),
            "participation_ratio": self._participation_from_svd(svd_values, clean.shape),
            "fractal_dim": self.fractal_dimension(clean),
            "coherence_max": coherence_max,
            "coherence_mean": coherence_mean,
            "spectral_flatness_db": self.spectral_flatness(clean),
            "dominant_frequency_bin": float(self.dominant_frequency(clean)),
        }
        self._cache[cache_key] = result
        return result

    def _svd_safe(self, matrix: np.ndarray) -> Optional[np.ndarray]:
        """Compute singular values once, reused by rank and participation ratio."""
        if matrix.size == 0 or min(matrix.shape) < 2:
            return None
        try:
            return np.linalg.svd(matrix, compute_uv=False)
        except np.linalg.LinAlgError:
            return None

    def _effective_rank_from_svd(
        self,
        svd_values: Optional[np.ndarray],
        shape: Tuple[int, ...],
    ) -> float:
        if svd_values is None or svd_values.size == 0:
            return 1.0
        total = float(np.sum(svd_values))
        if total <= self._metrics_cfg.norm_epsilon:
            return 1.0
        p = svd_values / total
        p = p[p > self._metrics_cfg.norm_epsilon]
        if p.size == 0:
            return 1.0
        h = -float(np.sum(p * np.log(p)))
        return float(math.exp(h))

    def _participation_from_svd(
        self,
        svd_values: Optional[np.ndarray],
        shape: Tuple[int, ...],
    ) -> float:
        if svd_values is None or svd_values.size == 0:
            return 1.0
        s2 = svd_values**2
        denom = float(np.sum(s2))
        if denom <= self._metrics_cfg.norm_epsilon:
            return 1.0
        return float((float(np.sum(svd_values)) ** 2) / denom / max(1, min(shape)))

    def sparsity(self, matrix: np.ndarray, threshold: float = 1e-6) -> float:
        """Fraction of entries whose absolute value is below ``threshold``."""
        flat = matrix.reshape(-1)
        if flat.size == 0:
            return 0.0
        return float(np.mean(np.abs(flat) < threshold))

    def entropy(self, matrix: np.ndarray) -> float:
        """Shannon entropy (nats) of the discrete value histogram.

        Uses probability mass (not density), which guarantees a non-negative
        result bounded above by ``log(histogram_bins)``. This is the standard
        convention for interpretability dashboards.
        """
        flat = matrix.reshape(-1)
        if flat.size > self._sampling.max_elements_for_histogram:
            rng = np.random.default_rng(self._config.generation.random_state)
            flat = rng.choice(flat, size=self._sampling.max_elements_for_histogram, replace=False)
        if flat.size == 0 or np.allclose(flat, flat[0]):
            return 0.0
        counts, _ = np.histogram(flat, bins=self._metrics_cfg.histogram_bins)
        total = counts.sum()
        if total == 0:
            return 0.0
        probs = counts.astype(np.float64) / float(total)
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs + self._metrics_cfg.entropy_epsilon)))

    def effective_rank(self, matrix: np.ndarray) -> float:
        """Effective rank via the exponential of the entropy of singular values."""
        mat = self._subsample_for_svd(matrix)
        return self._effective_rank_from_svd(self._svd_safe(mat), mat.shape)

    def participation_ratio(self, matrix: np.ndarray) -> float:
        """Participation ratio of the singular value spectrum.

        Defined as ``(sum s)^2 / sum(s^2)`` which equals the effective number
        of nonzero singular directions.
        """
        mat = self._subsample_for_svd(matrix)
        return self._participation_from_svd(self._svd_safe(mat), mat.shape)

    def fractal_dimension(self, matrix: np.ndarray) -> float:
        """Approximate the effective embedding dimension via PCA.

        Counts the number of principal components whose explained variance
        exceeds ``default_fractal_variance_floor``.
        """
        mat = self._subsample_for_pca(matrix)
        if min(mat.shape) < 2:
            return 1.0
        n_components = min(50, mat.shape[0], mat.shape[1])
        try:
            pca = PCA(n_components=n_components, random_state=self._config.generation.random_state)
            pca.fit(mat)
        except Exception:
            return 1.0
        evr = pca.explained_variance_ratio_
        floor = self._sampling.default_fractal_variance_floor
        return float(max(1.0, int(np.sum(evr > floor))))

    def coherence(self, matrix: np.ndarray) -> Tuple[float, float]:
        """Maximum and mean absolute cosine similarity between rows."""
        if matrix.ndim != 2 or matrix.shape[0] < 2:
            return 0.0, 0.0
        cap = self._sampling.max_rows_for_correlation
        sample = matrix[:cap] if matrix.shape[0] > cap else matrix
        norms = np.linalg.norm(sample, axis=1, keepdims=True)
        norms = np.where(norms < self._metrics_cfg.norm_epsilon, 1.0, norms)
        unit = sample / norms
        gram = np.abs(unit @ unit.T)
        np.fill_diagonal(gram, self._metrics_cfg.coherence_diagonal_value)
        if gram.size == 0:
            return 0.0, 0.0
        return float(np.max(gram)), float(np.mean(gram))

    def spectral_flatness(self, matrix: np.ndarray) -> float:
        """Spectral flatness (Wiener entropy) in dB averaged over rows."""
        if matrix.ndim != 2:
            return 0.0
        rows = matrix[: self._sampling.max_fft_rows]
        cap_cols = self._sampling.max_fft_samples_per_row
        if rows.shape[1] > cap_cols:
            rows = rows[:, :cap_cols]
        if rows.size == 0:
            return 0.0
        mags = np.abs(scipy_fft.rfft(rows, axis=1))
        mags = np.maximum(mags, self._metrics_cfg.spectral_flatness_epsilon)
        geo = np.exp(np.mean(np.log(mags), axis=1))
        arith = np.mean(mags, axis=1)
        ratio = geo / np.maximum(arith, self._metrics_cfg.spectral_flatness_epsilon)
        ratio = np.clip(ratio, self._metrics_cfg.spectral_flatness_epsilon, 1.0)
        return float(10.0 * np.mean(np.log10(ratio)))

    def dominant_frequency(self, matrix: np.ndarray) -> int:
        """Index of the dominant non-zero frequency bin of row 0."""
        if matrix.ndim != 2 or matrix.shape[1] < 2:
            return 0
        row = matrix[0, : self._sampling.max_fft_samples_per_row]
        mags = np.abs(scipy_fft.rfft(row))
        if mags.size <= 1:
            return 0
        return int(np.argmax(mags[1:]) + 1)

    def _subsample_for_svd(self, matrix: np.ndarray) -> np.ndarray:
        rows = self._sampling.max_rows_for_pca
        cols = self._sampling.max_cols_for_pca
        sub = matrix[:rows, :cols]
        return np.nan_to_num(sub, nan=0.0, posinf=0.0, neginf=0.0)

    def _subsample_for_pca(self, matrix: np.ndarray) -> np.ndarray:
        rows = self._sampling.max_rows_for_pca
        cols = self._sampling.max_cols_for_pca
        return np.nan_to_num(matrix[:rows, :cols], nan=0.0, posinf=0.0, neginf=0.0)


class FigureStyler:
    """Applies consistent styling to Plotly figures."""

    def __init__(self, config: ExplorerConfig) -> None:
        self._config = config

    def style_3d(self, fig: go.Figure, title: str) -> go.Figure:
        """Apply the standard 3D scene styling."""
        cfg = self._config
        pt = cfg.plot
        fig.update_layout(
            template=pt.template,
            title=dict(text=title, font=dict(size=18, color=cfg.theme.accent_primary)),
            height=cfg.default_figure_height,
            margin=dict(l=0, r=0, b=0, t=45),
            paper_bgcolor=pt.paper_bgcolor,
            plot_bgcolor=pt.plot_bgcolor,
            font=dict(color=pt.font_color),
            scene=dict(
                xaxis=dict(showbackground=False, gridcolor=pt.grid_color, zerolinecolor=pt.grid_color),
                yaxis=dict(showbackground=False, gridcolor=pt.grid_color, zerolinecolor=pt.grid_color),
                zaxis=dict(showbackground=False, gridcolor=pt.grid_color, zerolinecolor=pt.grid_color),
                aspectmode="cube",
            ),
            hoverlabel=dict(bgcolor="rgba(30, 45, 80, 0.9)", font_color=pt.font_color),
        )
        return fig

    def style_2d(self, fig: go.Figure, title: str, height: Optional[int] = None) -> go.Figure:
        """Apply the standard 2D figure styling."""
        cfg = self._config
        pt = cfg.plot
        fig.update_layout(
            template=pt.template,
            title=dict(text=title, font=dict(size=17, color=cfg.theme.accent_primary)),
            height=height or cfg.medium_figure_height,
            margin=dict(l=50, r=20, t=55, b=50),
            paper_bgcolor=pt.paper_bgcolor,
            plot_bgcolor=pt.plot_bgcolor,
            font=dict(color=pt.font_color),
            hoverlabel=dict(bgcolor="rgba(30, 45, 80, 0.9)", font_color=pt.font_color),
        )
        fig.update_xaxes(gridcolor=pt.grid_color, zerolinecolor=pt.grid_color)
        fig.update_yaxes(gridcolor=pt.grid_color, zerolinecolor=pt.grid_color)
        return fig


class VisualizationContext(Protocol):
    """Context passed to visualizers at render time."""

    inventory: TensorInventory
    projector: TensorProjector
    metrics: MetricCalculator
    styler: FigureStyler
    config: ExplorerConfig


@dataclass
class RenderContext:
    """Concrete visualization context implementation."""

    inventory: TensorInventory
    projector: TensorProjector
    metrics: MetricCalculator
    styler: FigureStyler
    config: ExplorerConfig


class BaseVisualizer(ABC):
    """Abstract base class for all visualizers.

    Subclasses implement :meth:`render` and :meth:`is_applicable`. The registry
    will instantiate them only when :meth:`is_applicable` returns ``True``.
    """

    label: str = ""
    description: str = ""

    def __init__(self, context: RenderContext) -> None:
        self._ctx = context

    @property
    def ctx(self) -> RenderContext:
        """Return the render context bound to this visualizer."""
        return self._ctx

    def is_applicable(self) -> bool:
        """Return ``True`` if there is data in the inventory to render."""
        return True

    @abstractmethod
    def render(self) -> None:
        """Render the visualizer into the current Streamlit container."""


class OverviewVisualizer(BaseVisualizer):
    """Top-level summary of the checkpoint: counts, roles, and inventory table."""

    label = "Overview"
    description = "Global inventory of parameters, roles, and layer composition."

    def render(self) -> None:
        rows = self._ctx.inventory.summary_rows()
        total = self._ctx.inventory.total_parameters()
        role_counts: Dict[str, int] = {}
        role_params: Dict[str, int] = {}
        for r in rows:
            role_counts[r["role"]] = role_counts.get(r["role"], 0) + 1
            role_params[r["role"]] = role_params.get(r["role"], 0) + r["numel"]

        self._render_headline(total, len(rows), len(self._ctx.inventory.layers()))
        self._render_role_breakdown(role_counts, role_params)
        self._render_inventory_table(rows)

    def _render_headline(self, total_params: int, num_tensors: int, num_layers: int) -> None:
        cols = st.columns(4)
        cols[0].metric("Total parameters", f"{total_params:,}")
        cols[1].metric("Distinct tensors", f"{num_tensors}")
        cols[2].metric("Transformer layers", f"{num_layers}")
        cols[3].metric("Bytes (fp32)", f"{total_params * 4 / (1024 ** 2):.2f} MiB")

    def _render_role_breakdown(
        self,
        role_counts: Dict[str, int],
        role_params: Dict[str, int],
    ) -> None:
        roles = sorted(role_counts.keys(), key=lambda r: -role_params[r])
        counts = [role_counts[r] for r in roles]
        params = [role_params[r] for r in roles]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Tensor count per role", "Parameter budget per role"),
            specs=[[{"type": "bar"}, {"type": "bar"}]],
        )
        fig.add_trace(
            go.Bar(x=roles, y=counts, marker_color=self._ctx.config.theme.accent_secondary),
            row=1, col=1,
        )
        fig.add_trace(
            go.Bar(x=roles, y=params, marker_color=self._ctx.config.theme.accent_primary),
            row=1, col=2,
        )
        fig.update_layout(showlegend=False)
        self._ctx.styler.style_2d(fig, "Checkpoint composition", height=self._ctx.config.medium_figure_height)
        fig.update_xaxes(tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)

    def _render_inventory_table(self, rows: List[Dict[str, Any]]) -> None:
        st.markdown("### Full parameter inventory")
        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
            column_config={
                "numel": st.column_config.NumberColumn(format="%d"),
                "mean": st.column_config.NumberColumn(format="%.4e"),
                "std": st.column_config.NumberColumn(format="%.4e"),
                "abs_max": st.column_config.NumberColumn(format="%.4e"),
            },
        )


class TensorExplorerVisualizer(BaseVisualizer):
    """Deep-dive into a single tensor: 3D cloud, heatmap, histograms, spectrum."""

    label = "Tensor Explorer"
    description = "Per-tensor 3D point cloud, heatmap, distribution, and spectral analysis."

    def render(self) -> None:
        names = self._ctx.inventory.names()
        if not names:
            st.info("No tensors present in the checkpoint.")
            return
        name = st.selectbox(
            "Tensor",
            options=names,
            help="Pick a parameter tensor to analyze.",
        )
        meta = self._ctx.inventory.meta(name)
        tensor = self._ctx.inventory.tensor(name)

        self._render_meta(meta)
        matrix = self._ctx.projector.to_matrix(tensor)
        subsample = self._ctx.projector.subsample(matrix)

        metrics = self._ctx.metrics.compute_all(subsample)
        self._render_metric_grid(metrics)

        projection_method = st.radio(
            "3D projection method",
            options=["pca", "random"],
            horizontal=True,
            help="PCA captures variance. Random is a Johnson-Lindenstrauss preserving projection.",
        )
        self._render_point_cloud(subsample, name, projection_method)

        cols = st.columns(2)
        with cols[0]:
            self._render_heatmap(subsample, name)
        with cols[1]:
            self._render_distribution(matrix, name)

        self._render_spectrum(subsample, name)
        self._render_singular_spectrum(subsample, name)

    def _render_meta(self, meta: Dict[str, Any]) -> None:
        cols = st.columns(5)
        cols[0].metric("Role", meta["role"])
        cols[1].metric("Layer", str(meta["layer"]) if meta["layer"] is not None else "-")
        cols[2].metric("Component", meta["component"] or "-")
        cols[3].metric("Shape", str(meta["shape"]))
        cols[4].metric("Elements", f"{meta['numel']:,}")

    def _render_metric_grid(self, metrics: Dict[str, float]) -> None:
        row1 = st.columns(4)
        row1[0].metric("Mean", f"{metrics['mean']:.3e}")
        row1[1].metric("Std", f"{metrics['std']:.3e}")
        row1[2].metric("|Max|", f"{metrics['abs_max']:.3e}")
        row1[3].metric("Frobenius norm", f"{metrics['frobenius_norm']:.3e}")

        row2 = st.columns(4)
        row2[0].metric("Entropy", f"{metrics['entropy']:.3f}")
        row2[1].metric("Effective rank", f"{metrics['effective_rank']:.2f}")
        row2[2].metric("Participation ratio", f"{metrics['participation_ratio']:.3f}")
        row2[3].metric("Coherence (max)", f"{metrics['coherence_max']:.3f}")

        row3 = st.columns(4)
        row3[0].metric("Coherence (mean)", f"{metrics['coherence_mean']:.3f}")
        row3[1].metric("Sparsity", f"{metrics['sparsity']*100:.2f} %")
        row3[2].metric("Fractal dim", f"{metrics['fractal_dim']:.1f}")
        row3[3].metric("Spectral flatness (dB)", f"{metrics['spectral_flatness_db']:.2f}")

    def _render_point_cloud(self, matrix: np.ndarray, name: str, method: str) -> None:
        emb, info = self._ctx.projector.project_3d(matrix, method=method)
        norms = np.linalg.norm(matrix, axis=1)
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=emb[:, 0],
                    y=emb[:, 1],
                    z=emb[:, 2],
                    mode="markers",
                    marker=dict(
                        size=self._ctx.config.point_cloud_marker_size,
                        color=norms,
                        colorscale=self._ctx.config.plot.sequential_scale,
                        opacity=self._ctx.config.point_cloud_opacity,
                        colorbar=dict(title="row L2 norm"),
                    ),
                    hovertemplate=(
                        "row %{customdata[0]}<br>"
                        "x=%{x:.4f} y=%{y:.4f} z=%{z:.4f}<br>"
                        "norm=%{customdata[1]:.4f}<extra></extra>"
                    ),
                    customdata=np.column_stack([np.arange(emb.shape[0]), norms]),
                )
            ]
        )
        explained = (
            f" (cumulative EVR {info.get('explained_variance_ratio', 0)*100:.1f}%)"
            if method == "pca"
            else ""
        )
        self._ctx.styler.style_3d(fig, f"{name} - 3D {method.upper()}{explained}")
        st.plotly_chart(fig, use_container_width=True)

    def _render_heatmap(self, matrix: np.ndarray, name: str) -> None:
        cap = self._ctx.config.sampling.max_elements_heatmap
        side = int(math.sqrt(cap))
        rows_cap = min(matrix.shape[0], side)
        cols_cap = min(matrix.shape[1], side)
        view = matrix[:rows_cap, :cols_cap]
        fig = go.Figure(
            data=go.Heatmap(
                z=view,
                colorscale=self._ctx.config.plot.diverging_scale,
                zmid=0.0,
                colorbar=dict(title="value"),
                hovertemplate="row=%{y} col=%{x}<br>value=%{z:.4f}<extra></extra>",
            )
        )
        self._ctx.styler.style_2d(
            fig,
            f"{name} - heatmap ({rows_cap}x{cols_cap})",
            height=self._ctx.config.medium_figure_height,
        )
        st.plotly_chart(fig, use_container_width=True)

    def _render_distribution(self, matrix: np.ndarray, name: str) -> None:
        flat = matrix.reshape(-1)
        cap = self._ctx.config.sampling.max_elements_for_histogram
        if flat.size > cap:
            rng = np.random.default_rng(self._ctx.config.generation.random_state)
            flat = rng.choice(flat, size=cap, replace=False)
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=flat,
                nbinsx=self._ctx.config.metrics.histogram_bins,
                marker_color=self._ctx.config.theme.accent_primary,
                opacity=0.85,
                name="value density",
            )
        )
        fig.add_vline(x=0.0, line_dash="dash", line_color="rgba(255,255,255,0.4)")
        self._ctx.styler.style_2d(
            fig,
            f"{name} - value distribution",
            height=self._ctx.config.medium_figure_height,
        )
        st.plotly_chart(fig, use_container_width=True)

    def _render_spectrum(self, matrix: np.ndarray, name: str) -> None:
        rows = matrix[: self._ctx.config.sampling.max_fft_rows]
        cap_cols = self._ctx.config.sampling.max_fft_samples_per_row
        if rows.shape[1] > cap_cols:
            rows = rows[:, :cap_cols]
        if rows.size == 0 or rows.shape[1] < 2:
            return
        freqs = scipy_fft.rfftfreq(rows.shape[1])
        fig = go.Figure()
        display_rows = min(6, rows.shape[0])
        for i in range(display_rows):
            mag = np.abs(scipy_fft.rfft(rows[i]))
            fig.add_trace(
                go.Scatter(
                    x=freqs,
                    y=mag,
                    mode="lines",
                    name=f"row {i}",
                    line=dict(width=2.1),
                    fill="tozeroy" if i == 0 else None,
                )
            )
        fig.update_xaxes(title="normalized frequency")
        fig.update_yaxes(title="magnitude")
        self._ctx.styler.style_2d(
            fig,
            f"{name} - row-wise spectra",
            height=self._ctx.config.medium_figure_height,
        )
        st.plotly_chart(fig, use_container_width=True)

    def _render_singular_spectrum(self, matrix: np.ndarray, name: str) -> None:
        if min(matrix.shape) < 2:
            return
        try:
            s = np.linalg.svd(matrix, compute_uv=False)
        except np.linalg.LinAlgError:
            return
        s = s[s > 0]
        if s.size == 0:
            return
        cumulative = np.cumsum(s**2) / np.sum(s**2)
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("singular values (log)", "cumulative variance"),
        )
        fig.add_trace(
            go.Scatter(
                x=np.arange(1, s.size + 1),
                y=s,
                mode="lines+markers",
                line=dict(color=self._ctx.config.theme.accent_primary, width=2.4),
                marker=dict(size=5),
                name="sigma_i",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=np.arange(1, s.size + 1),
                y=cumulative,
                mode="lines",
                line=dict(color=self._ctx.config.theme.accent_secondary, width=2.8),
                name="cumulative",
                fill="tozeroy",
            ),
            row=1, col=2,
        )
        fig.update_yaxes(type="log", row=1, col=1)
        fig.update_layout(showlegend=False)
        self._ctx.styler.style_2d(
            fig,
            f"{name} - singular spectrum",
            height=self._ctx.config.medium_figure_height,
        )
        st.plotly_chart(fig, use_container_width=True)


class QuaternionDecompositionVisualizer(BaseVisualizer):
    """Visualize QuaternionLinear layers by their (Ww, Wx, Wy, Wz) components."""

    label = "Quaternion Decomposition"
    description = "Inspect w/x/y/z components of quaternion linear layers."

    def is_applicable(self) -> bool:
        names = self._ctx.inventory.names()
        return any(self._ctx.inventory.meta(n)["component"] is not None for n in names)

    def render(self) -> None:
        bundles = self._group_quaternion_bundles()
        if not bundles:
            st.info("No quaternion-structured tensors detected.")
            return
        bundle_key = st.selectbox("Quaternion bundle", options=sorted(bundles.keys()))
        components = bundles[bundle_key]
        self._render_component_stats(components)
        self._render_component_heatmaps(components)
        self._render_component_spectra(components)
        self._render_unit_norm_distribution(components)

    def _group_quaternion_bundles(self) -> Dict[str, Dict[str, torch.Tensor]]:
        bundles: Dict[str, Dict[str, torch.Tensor]] = {}
        for name in self._ctx.inventory.names():
            meta = self._ctx.inventory.meta(name)
            comp = meta["component"]
            if comp is None:
                continue
            key = self._quaternion_bundle_key(name, comp)
            if key is None:
                continue
            bundles.setdefault(key, {})[comp] = self._ctx.inventory.tensor(name)
        return {k: v for k, v in bundles.items() if len(v) == 4}

    @staticmethod
    def _quaternion_bundle_key(name: str, component: str) -> Optional[str]:
        patterns = [
            (re.compile(rf"(^|\.)W{component}\.weight$"), ".W?.weight"),
            (re.compile(rf"(^|\.)W{component}\.bias$"), ".W?.bias"),
            (re.compile(rf"(^|\.)kr_{component}$"), ".kr_?"),
            (re.compile(rf"(^|\.)ki_{component}$"), ".ki_?"),
            (re.compile(rf"(^|[._]){component}_(kr|ki)$"), "._?_k?"),
        ]
        for pattern, template in patterns:
            m = pattern.search(name)
            if m:
                template_re = template.replace("?", component)
                return name.replace(template_re, template)
        return None

    def _render_component_stats(self, components: Dict[str, torch.Tensor]) -> None:
        cols = st.columns(4)
        for i, comp in enumerate(self._ctx.config.quaternion_component_names):
            if comp not in components:
                continue
            arr = components[comp].detach().cpu().numpy()
            with cols[i]:
                st.markdown(f"#### component `{comp}`")
                st.metric("Mean", f"{float(np.mean(arr)):.3e}")
                st.metric("Std", f"{float(np.std(arr)):.3e}")
                st.metric("|Max|", f"{float(np.max(np.abs(arr))):.3e}")

    def _render_component_heatmaps(self, components: Dict[str, torch.Tensor]) -> None:
        comps = [c for c in self._ctx.config.quaternion_component_names if c in components]
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f"component {c}" for c in comps],
            horizontal_spacing=0.08,
            vertical_spacing=0.12,
        )
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        for (r, c), comp in zip(positions, comps):
            mat = self._ctx.projector.to_matrix(components[comp])
            mat = self._ctx.projector.subsample(mat)
            fig.add_trace(
                go.Heatmap(
                    z=mat,
                    colorscale=self._ctx.config.plot.diverging_scale,
                    zmid=0.0,
                    showscale=(r == 1 and c == 2),
                    hovertemplate="row=%{y} col=%{x}<br>value=%{z:.4f}<extra></extra>",
                ),
                row=r, col=c,
            )
        self._ctx.styler.style_2d(fig, "Quaternion component heatmaps", height=self._ctx.config.default_figure_height)
        st.plotly_chart(fig, use_container_width=True)

    def _render_component_spectra(self, components: Dict[str, torch.Tensor]) -> None:
        fig = go.Figure()
        for comp, color in zip(
            self._ctx.config.quaternion_component_names,
            self._ctx.config.quaternion_component_colors,
        ):
            if comp not in components:
                continue
            arr = self._ctx.projector.to_matrix(components[comp])
            rows = arr[: self._ctx.config.sampling.max_fft_rows]
            cap_cols = self._ctx.config.sampling.max_fft_samples_per_row
            if rows.shape[1] > cap_cols:
                rows = rows[:, :cap_cols]
            if rows.size == 0 or rows.shape[1] < 2:
                continue
            mags = np.abs(scipy_fft.rfft(rows, axis=1)).mean(axis=0)
            freqs = scipy_fft.rfftfreq(rows.shape[1])
            fig.add_trace(
                go.Scatter(
                    x=freqs,
                    y=mags,
                    mode="lines",
                    name=f"component {comp}",
                    line=dict(color=color, width=2.4),
                )
            )
        fig.update_xaxes(title="normalized frequency")
        fig.update_yaxes(title="mean magnitude")
        self._ctx.styler.style_2d(fig, "Component spectra (row-averaged)", height=self._ctx.config.medium_figure_height)
        st.plotly_chart(fig, use_container_width=True)

    def _render_unit_norm_distribution(self, components: Dict[str, torch.Tensor]) -> None:
        comps = [c for c in self._ctx.config.quaternion_component_names if c in components]
        if len(comps) != 4:
            return
        arrays = [self._ctx.projector.to_matrix(components[c]) for c in comps]
        min_rows = min(a.shape[0] for a in arrays)
        min_cols = min(a.shape[1] for a in arrays)
        stacks = np.stack([a[:min_rows, :min_cols] for a in arrays], axis=-1)
        norms = np.linalg.norm(stacks, axis=-1).reshape(-1)
        rng = np.random.default_rng(self._ctx.config.generation.random_state)
        cap = self._ctx.config.sampling.max_elements_for_histogram
        if norms.size > cap:
            norms = rng.choice(norms, size=cap, replace=False)
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=norms,
                nbinsx=self._ctx.config.metrics.histogram_bins,
                marker_color=self._ctx.config.theme.accent_primary,
                name="quaternion norm",
            )
        )
        fig.add_vline(
            x=1.0,
            line_dash="dash",
            line_color="rgba(255,120,120,0.8)",
            annotation_text="unit",
            annotation_position="top right",
        )
        fig.update_xaxes(title="||q||")
        fig.update_yaxes(title="count")
        self._ctx.styler.style_2d(
            fig,
            "Quaternion norm distribution (sqrt(w^2+x^2+y^2+z^2))",
            height=self._ctx.config.medium_figure_height,
        )
        st.plotly_chart(fig, use_container_width=True)


class SpectralKernelVisualizer(BaseVisualizer):
    """Visualize complex spectral kernels (kr/ki pairs) in the frequency plane."""

    label = "Spectral Kernels"
    description = "Magnitude and phase of learned complex spectral kernels."

    def is_applicable(self) -> bool:
        return any(
            self._ctx.inventory.meta(n)["is_complex_kernel"]
            for n in self._ctx.inventory.names()
        )

    def render(self) -> None:
        pairs = self._pair_kr_ki()
        if not pairs:
            st.info("No complex spectral kernel pairs (kr/ki) were found.")
            return
        key = st.selectbox("Spectral kernel pair", options=sorted(pairs.keys()))
        kr, ki = pairs[key]
        complex_kernel = kr.numpy() + 1j * ki.numpy()
        magnitude = np.abs(complex_kernel)
        phase = np.angle(complex_kernel)

        mag2d, phase2d = self._reshape_to_2d(magnitude, phase)
        self._render_magnitude_phase(mag2d, phase2d, key)
        self._render_complex_scatter(complex_kernel, key)
        self._render_radial_profile(mag2d, key)

    def _pair_kr_ki(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        pairs: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {}
        for name in self._ctx.inventory.names():
            meta = self._ctx.inventory.meta(name)
            if not meta["is_complex_kernel"]:
                continue
            base_key = re.sub(r"\bkr\b|\bki\b|kr_[wxyz]|ki_[wxyz]", "K", name)
            slot = pairs.setdefault(base_key, (None, None))
            kr_slot, ki_slot = slot
            if "kr" in name:
                pairs[base_key] = (self._ctx.inventory.tensor(name), ki_slot)
            elif "ki" in name:
                pairs[base_key] = (kr_slot, self._ctx.inventory.tensor(name))
        return {k: (r, i) for k, (r, i) in pairs.items() if r is not None and i is not None}

    def _reshape_to_2d(
        self, magnitude: np.ndarray, phase: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if magnitude.ndim == 1:
            return magnitude.reshape(1, -1), phase.reshape(1, -1)
        if magnitude.ndim == 2:
            return magnitude, phase
        leading = magnitude.shape[0]
        trailing = int(np.prod(magnitude.shape[1:]))
        return magnitude.reshape(leading, trailing), phase.reshape(leading, trailing)

    def _render_magnitude_phase(
        self, magnitude: np.ndarray, phase: np.ndarray, key: str
    ) -> None:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("|K(f)| magnitude", "arg K(f) phase"),
            horizontal_spacing=0.08,
        )
        fig.add_trace(
            go.Heatmap(
                z=magnitude,
                colorscale=self._ctx.config.plot.spectral_scale,
                colorbar=dict(title="|K|", x=0.45),
                hovertemplate="row=%{y} col=%{x}<br>|K|=%{z:.4f}<extra></extra>",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Heatmap(
                z=phase,
                colorscale="Twilight",
                zmin=-math.pi,
                zmax=math.pi,
                colorbar=dict(title="arg K", x=1.0),
                hovertemplate="row=%{y} col=%{x}<br>arg=%{z:.3f}<extra></extra>",
            ),
            row=1, col=2,
        )
        self._ctx.styler.style_2d(fig, f"{key} - complex kernel", height=self._ctx.config.medium_figure_height)
        st.plotly_chart(fig, use_container_width=True)

    def _render_complex_scatter(self, complex_kernel: np.ndarray, key: str) -> None:
        flat = complex_kernel.reshape(-1)
        cap = self._ctx.config.sampling.max_elements_for_histogram
        if flat.size > cap:
            rng = np.random.default_rng(self._ctx.config.generation.random_state)
            idx = rng.choice(flat.size, size=cap, replace=False)
            flat = flat[idx]
        magnitudes = np.abs(flat)
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=flat.real,
                    y=flat.imag,
                    mode="markers",
                    marker=dict(
                        size=3,
                        color=magnitudes,
                        colorscale=self._ctx.config.plot.spectral_scale,
                        opacity=0.7,
                        colorbar=dict(title="|K|"),
                    ),
                    hovertemplate="Re=%{x:.4f}<br>Im=%{y:.4f}<extra></extra>",
                )
            ]
        )
        fig.update_xaxes(title="Re(K)", zerolinecolor="rgba(255,255,255,0.3)")
        fig.update_yaxes(title="Im(K)", zerolinecolor="rgba(255,255,255,0.3)", scaleanchor="x", scaleratio=1.0)
        self._ctx.styler.style_2d(
            fig,
            f"{key} - complex plane",
            height=self._ctx.config.default_figure_height,
        )
        st.plotly_chart(fig, use_container_width=True)

    def _render_radial_profile(self, magnitude: np.ndarray, key: str) -> None:
        h, w = magnitude.shape
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        yy, xx = np.indices((h, w))
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        r_int = r.astype(np.int32)
        max_r = int(r_int.max()) + 1
        means = np.zeros(max_r)
        counts = np.zeros(max_r)
        np.add.at(means, r_int, magnitude)
        np.add.at(counts, r_int, 1)
        counts = np.maximum(counts, 1)
        profile = means / counts
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=np.arange(max_r),
                y=profile,
                mode="lines+markers",
                line=dict(color=self._ctx.config.theme.accent_primary, width=2.5),
                fill="tozeroy",
                marker=dict(size=5),
            )
        )
        fig.update_xaxes(title="radial bin")
        fig.update_yaxes(title="mean |K|")
        self._ctx.styler.style_2d(
            fig,
            f"{key} - radial magnitude profile",
            height=self._ctx.config.medium_figure_height,
        )
        st.plotly_chart(fig, use_container_width=True)


class TorusTopologyVisualizer(BaseVisualizer):
    """3D torus graph of the QuaternionTorusBrain node embeddings and edges."""

    label = "Torus Topology"
    description = "3D graph of the learned torus brain, edges colored by quaternion rotation."

    def is_applicable(self) -> bool:
        return bool(self._ctx.inventory.filter(role="torus_node_embedding"))

    def render(self) -> None:
        node_names = self._ctx.inventory.filter(role="torus_node_embedding")
        edge_names = self._ctx.inventory.filter(role="torus_edge_quaternion")
        if not node_names:
            st.info("No torus node embeddings found.")
            return
        node_name = st.selectbox("Node embedding tensor", options=node_names)
        edge_name = edge_names[0] if edge_names else None

        nodes = self._ctx.inventory.tensor(node_name).numpy()
        edges = self._ctx.inventory.tensor(edge_name).numpy() if edge_name else None

        n_nodes = nodes.shape[0]
        radial_bins, angular_bins = self._infer_grid(n_nodes)

        self._render_headline_metrics(nodes, edges, radial_bins, angular_bins)
        self._render_3d_torus(nodes, edges, radial_bins, angular_bins)
        self._render_node_correlation(nodes)
        if edges is not None:
            self._render_edge_quaternions(edges)

    def _infer_grid(self, n_nodes: int) -> Tuple[int, int]:
        default_r = self._ctx.config.torus_radial_bins_default
        default_a = self._ctx.config.torus_angular_bins_default
        if n_nodes == default_r * default_a:
            return default_r, default_a
        for r in range(2, int(math.isqrt(n_nodes)) + 1):
            if n_nodes % r == 0:
                return r, n_nodes // r
        return 1, n_nodes

    def _render_headline_metrics(
        self,
        nodes: np.ndarray,
        edges: Optional[np.ndarray],
        radial_bins: int,
        angular_bins: int,
    ) -> None:
        cols = st.columns(4)
        cols[0].metric("Nodes", f"{nodes.shape[0]}")
        cols[1].metric("Grid", f"{radial_bins} x {angular_bins}")
        cols[2].metric("Embed dim", f"{nodes.shape[1]}")
        cols[3].metric("Edge types", f"{edges.shape[0] if edges is not None else 0}")

    def _render_3d_torus(
        self,
        nodes: np.ndarray,
        edges: Optional[np.ndarray],
        radial_bins: int,
        angular_bins: int,
    ) -> None:
        positions, node_colors = self._torus_positions(nodes, radial_bins, angular_bins)
        fig = go.Figure()
        self._add_edges(fig, positions, radial_bins, angular_bins, edges)
        self._add_nodes(fig, positions, node_colors, nodes.shape[0])
        self._ctx.styler.style_3d(fig, "Torus topology graph (embedded in 3D)")
        st.plotly_chart(fig, use_container_width=True)

    def _torus_positions(
        self,
        nodes: np.ndarray,
        radial_bins: int,
        angular_bins: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_nodes = nodes.shape[0]
        pca = PCA(n_components=1, random_state=self._ctx.config.generation.random_state)
        projected = pca.fit_transform(nodes).reshape(-1)
        projected = (projected - projected.min()) / max(
            projected.max() - projected.min(), self._ctx.config.metrics.norm_epsilon
        )
        major_radius = 2.0
        minor_radius = 0.9
        positions = np.zeros((n_nodes, 3))
        for idx in range(n_nodes):
            r = idx // angular_bins
            a = idx % angular_bins
            theta = 2.0 * math.pi * a / max(1, angular_bins)
            phi = 2.0 * math.pi * r / max(1, radial_bins)
            x = (major_radius + minor_radius * math.cos(phi)) * math.cos(theta)
            y = (major_radius + minor_radius * math.cos(phi)) * math.sin(theta)
            z = minor_radius * math.sin(phi)
            positions[idx] = (x, y, z)
        return positions, projected

    def _add_edges(
        self,
        fig: go.Figure,
        positions: np.ndarray,
        radial_bins: int,
        angular_bins: int,
        edges: Optional[np.ndarray],
    ) -> None:
        edge_colors = [
            "rgba(100, 181, 246, 0.85)",
            "rgba(240, 98, 146, 0.85)",
            "rgba(129, 199, 132, 0.85)",
            "rgba(255, 183, 77, 0.85)",
        ]
        segments = self._build_segments(positions, radial_bins, angular_bins)
        for edge_type, color in zip(range(4), edge_colors):
            rotations_norm = 0.0
            if edges is not None and edge_type < edges.shape[0]:
                rotations_norm = float(np.linalg.norm(edges[edge_type]))
            xs, ys, zs = [], [], []
            for s in segments[edge_type]:
                xs.extend([positions[s[0], 0], positions[s[1], 0], None])
                ys.extend([positions[s[0], 1], positions[s[1], 1], None])
                zs.extend([positions[s[0], 2], positions[s[1], 2], None])
            if not xs:
                continue
            label = self._edge_label(edge_type)
            fig.add_trace(
                go.Scatter3d(
                    x=xs, y=ys, z=zs,
                    mode="lines",
                    line=dict(color=color, width=self._ctx.config.graph_edge_width),
                    name=f"{label} (||q||={rotations_norm:.3f})",
                    hoverinfo="name",
                )
            )

    @staticmethod
    def _edge_label(edge_type: int) -> str:
        return {
            0: "angular-left",
            1: "angular-right",
            2: "radial-down",
            3: "radial-up",
        }.get(edge_type, f"edge-{edge_type}")

    @staticmethod
    def _build_segments(
        positions: np.ndarray,
        radial_bins: int,
        angular_bins: int,
    ) -> Dict[int, List[Tuple[int, int]]]:
        segments: Dict[int, List[Tuple[int, int]]] = {0: [], 1: [], 2: [], 3: []}
        for r in range(radial_bins):
            for a in range(angular_bins):
                i = r * angular_bins + a
                left = r * angular_bins + (a - 1) % angular_bins
                right = r * angular_bins + (a + 1) % angular_bins
                down = ((r - 1) % radial_bins) * angular_bins + a
                up = ((r + 1) % radial_bins) * angular_bins + a
                segments[0].append((i, left))
                segments[1].append((i, right))
                segments[2].append((i, down))
                segments[3].append((i, up))
        return segments

    def _add_nodes(
        self,
        fig: go.Figure,
        positions: np.ndarray,
        node_colors: np.ndarray,
        n_nodes: int,
    ) -> None:
        fig.add_trace(
            go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode="markers+text",
                marker=dict(
                    size=self._ctx.config.graph_node_size,
                    color=node_colors,
                    colorscale=self._ctx.config.plot.spectral_scale,
                    opacity=0.95,
                    line=dict(color="white", width=1),
                    colorbar=dict(title="PCA1 of embed"),
                ),
                text=[f"n{i}" for i in range(n_nodes)],
                textposition="top center",
                name="nodes",
                hovertemplate="node %{text}<br>x=%{x:.2f} y=%{y:.2f} z=%{z:.2f}<extra></extra>",
            )
        )

    def _render_node_correlation(self, nodes: np.ndarray) -> None:
        norms = np.linalg.norm(nodes, axis=1, keepdims=True)
        norms = np.where(norms < self._ctx.config.metrics.norm_epsilon, 1.0, norms)
        unit = nodes / norms
        cosine = unit @ unit.T
        fig = go.Figure(
            data=go.Heatmap(
                z=cosine,
                colorscale=self._ctx.config.plot.diverging_scale,
                zmid=0.0,
                zmin=-1.0,
                zmax=1.0,
                colorbar=dict(title="cos"),
            )
        )
        self._ctx.styler.style_2d(
            fig,
            "Node embedding cosine similarity",
            height=self._ctx.config.medium_figure_height,
        )
        st.plotly_chart(fig, use_container_width=True)

    def _render_edge_quaternions(self, edges: np.ndarray) -> None:
        labels = [self._edge_label(i) for i in range(edges.shape[0])]
        fig = go.Figure()
        comps = self._ctx.config.quaternion_component_names
        colors = self._ctx.config.quaternion_component_colors
        for ci, (comp, color) in enumerate(zip(comps, colors)):
            if ci >= edges.shape[1]:
                break
            fig.add_trace(
                go.Bar(
                    x=labels,
                    y=edges[:, ci],
                    name=f"component {comp}",
                    marker_color=color,
                )
            )
        fig.update_layout(barmode="group")
        fig.update_yaxes(title="value")
        self._ctx.styler.style_2d(
            fig,
            "Learned edge quaternions by component",
            height=self._ctx.config.medium_figure_height,
        )
        st.plotly_chart(fig, use_container_width=True)


class AttentionVisualizer(BaseVisualizer):
    """Per-layer attention Q/K/V/O projection analysis with per-head split."""

    label = "Attention Projections"
    description = "Per-head view of Q/K/V/O weight matrices by transformer layer."

    def is_applicable(self) -> bool:
        return any(
            self._ctx.inventory.meta(n)["role"] in {"attn_q", "attn_k", "attn_v", "attn_o"}
            for n in self._ctx.inventory.names()
        )

    def render(self) -> None:
        layers = self._ctx.inventory.layers()
        if not layers:
            st.info("No transformer layers were detected.")
            return
        layer = st.selectbox("Layer index", options=layers)
        proj = st.radio(
            "Projection",
            options=["attn_q", "attn_k", "attn_v", "attn_o"],
            horizontal=True,
            format_func=lambda s: s.split("_")[1].upper(),
        )
        names = self._ctx.inventory.filter(role=proj, layer=layer)
        if not names:
            st.info(f"No tensor with role={proj} at layer={layer}.")
            return
        name = names[0]
        tensor = self._ctx.inventory.tensor(name)
        matrix = self._ctx.projector.to_matrix(tensor)
        subsample = self._ctx.projector.subsample(matrix)

        self._render_per_head_norms(matrix, proj)
        self._render_per_head_spectrum(matrix, proj, layer)
        self._render_head_similarity(matrix, proj, layer)

        metrics = self._ctx.metrics.compute_all(subsample)
        self._render_summary_metrics(metrics)

    def _infer_head_count(self, matrix: np.ndarray, proj: str) -> Optional[int]:
        temperatures = self._ctx.inventory.filter(role="attn_temperature")
        n_heads_candidate = None
        for candidate in (4, 6, 8, 12, 16):
            if matrix.shape[0] % candidate == 0:
                n_heads_candidate = candidate
                break
        return n_heads_candidate

    def _render_per_head_norms(self, matrix: np.ndarray, proj: str) -> None:
        n_heads = self._infer_head_count(matrix, proj)
        if n_heads is None:
            st.warning("Cannot infer head count from shape; showing flat norms.")
            row_norms = np.linalg.norm(matrix, axis=1)
            n_heads = 1
        d_head = matrix.shape[0] // max(1, n_heads)
        reshaped = matrix[: n_heads * d_head].reshape(n_heads, d_head, -1)
        per_head_norm = np.linalg.norm(reshaped.reshape(n_heads, -1), axis=1)
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=[f"head {i}" for i in range(n_heads)],
                y=per_head_norm,
                marker_color=self._ctx.config.theme.accent_primary,
                text=[f"{v:.3f}" for v in per_head_norm],
                textposition="outside",
            )
        )
        fig.update_yaxes(title="Frobenius norm")
        self._ctx.styler.style_2d(
            fig,
            f"{proj.upper()} - per-head norms",
            height=self._ctx.config.medium_figure_height,
        )
        st.plotly_chart(fig, use_container_width=True)

    def _render_per_head_spectrum(self, matrix: np.ndarray, proj: str, layer: int) -> None:
        n_heads = self._infer_head_count(matrix, proj) or 1
        d_head = matrix.shape[0] // n_heads
        if d_head < 2:
            return
        fig = go.Figure()
        for h in range(min(n_heads, 12)):
            head_matrix = matrix[h * d_head : (h + 1) * d_head]
            try:
                s = np.linalg.svd(head_matrix, compute_uv=False)
            except np.linalg.LinAlgError:
                continue
            s = s[s > 0]
            fig.add_trace(
                go.Scatter(
                    x=np.arange(1, s.size + 1),
                    y=s,
                    mode="lines+markers",
                    name=f"head {h}",
                    line=dict(width=2),
                    marker=dict(size=4),
                )
            )
        fig.update_yaxes(type="log", title="singular value")
        fig.update_xaxes(title="index")
        self._ctx.styler.style_2d(
            fig,
            f"{proj.upper()} per-head singular spectra (layer {layer})",
            height=self._ctx.config.medium_figure_height,
        )
        st.plotly_chart(fig, use_container_width=True)

    def _render_head_similarity(self, matrix: np.ndarray, proj: str, layer: int) -> None:
        n_heads = self._infer_head_count(matrix, proj) or 1
        if n_heads <= 1:
            return
        d_head = matrix.shape[0] // n_heads
        flat_heads = matrix[: n_heads * d_head].reshape(n_heads, -1)
        norms = np.linalg.norm(flat_heads, axis=1, keepdims=True)
        norms = np.where(norms < self._ctx.config.metrics.norm_epsilon, 1.0, norms)
        unit = flat_heads / norms
        cosine = unit @ unit.T
        fig = go.Figure(
            data=go.Heatmap(
                z=cosine,
                colorscale=self._ctx.config.plot.diverging_scale,
                zmid=0.0,
                zmin=-1.0,
                zmax=1.0,
                colorbar=dict(title="cos"),
            )
        )
        self._ctx.styler.style_2d(
            fig,
            f"{proj.upper()} head-to-head similarity (layer {layer})",
            height=self._ctx.config.medium_figure_height,
        )
        st.plotly_chart(fig, use_container_width=True)

    def _render_summary_metrics(self, metrics: Dict[str, float]) -> None:
        cols = st.columns(4)
        cols[0].metric("Frobenius", f"{metrics['frobenius_norm']:.3e}")
        cols[1].metric("Effective rank", f"{metrics['effective_rank']:.2f}")
        cols[2].metric("Participation ratio", f"{metrics['participation_ratio']:.3f}")
        cols[3].metric("Spectral flatness (dB)", f"{metrics['spectral_flatness_db']:.2f}")


class MoEVisualizer(BaseVisualizer):
    """Router logit landscape and expert weight geometry."""

    label = "Mixture of Experts"
    description = "Router logits, expert utilization proxies, and expert weight geometry."

    def is_applicable(self) -> bool:
        return bool(self._ctx.inventory.filter(role="moe_router")) or bool(
            self._ctx.inventory.filter(role="moe_expert")
        )

    def render(self) -> None:
        layers = sorted(
            {
                self._ctx.inventory.meta(n)["layer"]
                for n in self._ctx.inventory.filter(role="moe_router")
                if self._ctx.inventory.meta(n)["layer"] is not None
            }
        )
        if not layers:
            st.info("No MoE routers detected; MoE may be disabled in this checkpoint.")
            return
        layer = st.selectbox("Transformer layer", options=layers)

        router_name = self._ctx.inventory.filter(role="moe_router", layer=layer)
        if not router_name:
            return
        router = self._ctx.inventory.tensor(router_name[0]).numpy()

        self._render_router_norms(router, layer)
        self._render_routing_probe(router, layer)
        self._render_expert_similarity(layer)

    def _render_router_norms(self, router: np.ndarray, layer: int) -> None:
        if router.ndim != 2:
            st.warning(f"Unexpected router shape: {router.shape}")
            return
        per_expert_norm = np.linalg.norm(router, axis=1)
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=[f"expert {i}" for i in range(router.shape[0])],
                y=per_expert_norm,
                marker_color=self._ctx.config.theme.accent_secondary,
                text=[f"{v:.3f}" for v in per_expert_norm],
                textposition="outside",
            )
        )
        fig.update_yaxes(title="||router row||_2")
        self._ctx.styler.style_2d(
            fig,
            f"Layer {layer} - router row norms (one per expert)",
            height=self._ctx.config.medium_figure_height,
        )
        st.plotly_chart(fig, use_container_width=True)

    def _render_routing_probe(self, router: np.ndarray, layer: int) -> None:
        rng = np.random.default_rng(self._ctx.config.generation.random_state)
        n_probe = self._ctx.config.generation.routing_probe_tokens
        d_model = router.shape[1]
        probe = rng.standard_normal(size=(n_probe, d_model)).astype(np.float32)
        logits = probe @ router.T
        probs = self._softmax(logits)
        mean_probs = probs.mean(axis=0)
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                "Mean routing probability per expert",
                f"Probability distribution across {n_probe} synthetic tokens",
            ),
            specs=[[{"type": "bar"}, {"type": "heatmap"}]],
        )
        fig.add_trace(
            go.Bar(
                x=[f"e{i}" for i in range(mean_probs.size)],
                y=mean_probs,
                marker_color=self._ctx.config.theme.accent_primary,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Heatmap(
                z=probs[: min(n_probe, 128)],
                colorscale=self._ctx.config.plot.spectral_scale,
                colorbar=dict(title="p"),
            ),
            row=1, col=2,
        )
        fig.update_xaxes(title="expert", row=1, col=1)
        fig.update_yaxes(title="p", row=1, col=1)
        fig.update_xaxes(title="expert", row=1, col=2)
        fig.update_yaxes(title="token", row=1, col=2)
        self._ctx.styler.style_2d(
            fig,
            f"Layer {layer} - synthetic Gaussian routing probe",
            height=self._ctx.config.medium_figure_height,
        )
        st.plotly_chart(fig, use_container_width=True)

    def _render_expert_similarity(self, layer: int) -> None:
        expert_names = [
            n for n in self._ctx.inventory.filter(role="moe_expert", layer=layer)
            if n.endswith("down_proj.weight")
        ]
        if len(expert_names) < 2:
            return
        vectors = []
        labels = []
        for name in expert_names:
            mat = self._ctx.inventory.tensor(name).numpy()
            vectors.append(mat.reshape(-1))
            parts = name.split(".")
            exp_idx = next((p for p in parts if p.isdigit()), "?")
            labels.append(f"e{exp_idx}")
        stacked = np.stack(vectors, axis=0)
        norms = np.linalg.norm(stacked, axis=1, keepdims=True)
        norms = np.where(norms < self._ctx.config.metrics.norm_epsilon, 1.0, norms)
        unit = stacked / norms
        cosine = unit @ unit.T
        fig = go.Figure(
            data=go.Heatmap(
                z=cosine,
                x=labels,
                y=labels,
                colorscale=self._ctx.config.plot.diverging_scale,
                zmid=0.0,
                zmin=-1.0,
                zmax=1.0,
                colorbar=dict(title="cos"),
            )
        )
        self._ctx.styler.style_2d(
            fig,
            f"Layer {layer} - expert down_proj cosine similarity",
            height=self._ctx.config.medium_figure_height,
        )
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=-1, keepdims=True)


class LayerEvolutionVisualizer(BaseVisualizer):
    """Track how metrics evolve across transformer layers for a chosen role."""

    label = "Layer Evolution"
    description = "Track weight statistics and spectral signatures across transformer depth."

    def is_applicable(self) -> bool:
        return len(self._ctx.inventory.layers()) > 1

    def render(self) -> None:
        roles = [
            r for r in self._ctx.inventory.roles()
            if any(self._ctx.inventory.meta(n)["layer"] is not None
                   for n in self._ctx.inventory.filter(role=r))
        ]
        if not roles:
            st.info("No layered tensors detected.")
            return
        role = st.selectbox("Role to track", options=roles)
        layers = self._ctx.inventory.layers()
        series: Dict[str, List[float]] = {
            "frobenius_norm": [],
            "effective_rank": [],
            "participation_ratio": [],
            "entropy": [],
            "spectral_flatness_db": [],
            "coherence_mean": [],
            "fractal_dim": [],
            "sparsity": [],
        }
        kept_layers: List[int] = []
        for layer in layers:
            names = self._ctx.inventory.filter(role=role, layer=layer)
            if not names:
                continue
            matrices = [
                self._ctx.projector.subsample(
                    self._ctx.projector.to_matrix(self._ctx.inventory.tensor(n))
                )
                for n in names
            ]
            combined = np.vstack([m.reshape(m.shape[0], -1) for m in matrices])
            m = self._ctx.metrics.compute_all(combined)
            for k in series:
                series[k].append(float(m[k]))
            kept_layers.append(layer)

        if not kept_layers:
            st.info(f"No tensors with role={role} at any layer.")
            return
        self._render_metric_grid(role, kept_layers, series)

    def _render_metric_grid(
        self,
        role: str,
        layers: List[int],
        series: Dict[str, List[float]],
    ) -> None:
        titles = list(series.keys())
        fig = make_subplots(
            rows=2, cols=4,
            subplot_titles=titles,
            horizontal_spacing=0.07,
            vertical_spacing=0.16,
        )
        positions = [
            (1, 1), (1, 2), (1, 3), (1, 4),
            (2, 1), (2, 2), (2, 3), (2, 4),
        ]
        for (r, c), title in zip(positions, titles):
            fig.add_trace(
                go.Scatter(
                    x=layers,
                    y=series[title],
                    mode="lines+markers",
                    line=dict(color=self._ctx.config.theme.accent_primary, width=2.4),
                    marker=dict(size=7),
                    showlegend=False,
                ),
                row=r, col=c,
            )
            fig.update_xaxes(title="layer", row=r, col=c)
        self._ctx.styler.style_2d(
            fig,
            f"Evolution across layers for role='{role}'",
            height=self._ctx.config.default_figure_height,
        )
        st.plotly_chart(fig, use_container_width=True)


class GlobalGeometryVisualizer(BaseVisualizer):
    """Cross-tensor geometry: embed every weight matrix as a 3D point via summary features."""

    label = "Global Geometry"
    description = "Embed every tensor as a point in a scientific feature space."

    def render(self) -> None:
        rows = self._ctx.inventory.summary_rows()
        progress = st.progress(0.0, text="Computing per-tensor features...")
        features: List[List[float]] = []
        labels: List[str] = []
        roles: List[str] = []
        sizes: List[int] = []
        total = len(rows)
        for idx, r in enumerate(rows):
            name = r["name"]
            tensor = self._ctx.inventory.tensor(name)
            mat = self._ctx.projector.to_matrix(tensor)
            sub = self._ctx.projector.subsample(
                mat,
                max_rows=self._ctx.config.sampling.global_scan_max_rows,
                max_cols=self._ctx.config.sampling.global_scan_max_cols,
            )
            m = self._ctx.metrics.compute_all(sub)
            features.append(
                [
                    math.log1p(max(0.0, m["frobenius_norm"])),
                    m["entropy"],
                    math.log1p(max(0.0, m["effective_rank"])),
                    m["coherence_mean"],
                    m["spectral_flatness_db"],
                    m["participation_ratio"],
                    math.log1p(max(0.0, m["fractal_dim"])),
                ]
            )
            labels.append(name)
            roles.append(r["role"])
            sizes.append(r["numel"])
            progress.progress((idx + 1) / max(1, total), text=f"Processed {idx + 1}/{total} tensors")
        progress.empty()
        X = np.array(features, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True) + self._ctx.config.metrics.norm_epsilon
        X_std = (X - mean) / std
        if X_std.shape[0] < 4:
            st.info("Need at least 4 tensors to build the global geometry view.")
            return
        pca = PCA(n_components=3, random_state=self._ctx.config.generation.random_state)
        emb = pca.fit_transform(X_std)
        self._render_scatter(emb, labels, roles, sizes, pca)
        self._render_feature_correlation(X_std, features[0])

    def _render_scatter(
        self,
        emb: np.ndarray,
        labels: List[str],
        roles: List[str],
        sizes: List[int],
        pca: PCA,
    ) -> None:
        fig = go.Figure()
        unique_roles = sorted(set(roles))
        palette = self._build_palette(len(unique_roles))
        for role, color in zip(unique_roles, palette):
            mask = [r == role for r in roles]
            idxs = [i for i, m in enumerate(mask) if m]
            if not idxs:
                continue
            role_sizes = np.array([sizes[i] for i in idxs])
            scaled = 6.0 + 14.0 * (np.log1p(role_sizes) / max(1.0, np.log1p(role_sizes.max())))
            fig.add_trace(
                go.Scatter3d(
                    x=emb[idxs, 0],
                    y=emb[idxs, 1],
                    z=emb[idxs, 2],
                    mode="markers",
                    marker=dict(
                        size=scaled,
                        color=color,
                        opacity=0.9,
                        line=dict(color="rgba(255,255,255,0.25)", width=1),
                    ),
                    name=role,
                    text=[labels[i] for i in idxs],
                    hovertemplate="%{text}<br>role=" + role + "<extra></extra>",
                )
            )
        evr_total = float(np.sum(pca.explained_variance_ratio_)) * 100
        self._ctx.styler.style_3d(
            fig,
            f"Global geometry (feature-space PCA, cum EVR {evr_total:.1f}%)",
        )
        st.plotly_chart(fig, use_container_width=True)

    def _render_feature_correlation(self, X_std: np.ndarray, example: List[float]) -> None:
        feature_names = [
            "log(||W||_F)",
            "entropy",
            "log(eff. rank)",
            "coherence mean",
            "spectral flatness (dB)",
            "participation ratio",
            "log(fractal dim)",
        ]
        if X_std.shape[1] != len(feature_names):
            return
        corr = np.corrcoef(X_std.T)
        fig = go.Figure(
            data=go.Heatmap(
                z=corr,
                x=feature_names,
                y=feature_names,
                colorscale=self._ctx.config.plot.diverging_scale,
                zmid=0.0,
                zmin=-1.0,
                zmax=1.0,
                colorbar=dict(title="corr"),
            )
        )
        self._ctx.styler.style_2d(
            fig,
            "Cross-tensor feature correlation",
            height=self._ctx.config.medium_figure_height,
        )
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def _build_palette(n: int) -> List[str]:
        base = [
            "#64b5f6", "#f06292", "#81c784", "#ffb74d", "#ba68c8",
            "#4db6ac", "#ff8a65", "#a1887f", "#90a4ae", "#aed581",
            "#7986cb", "#ffd54f", "#e57373", "#4dd0e1", "#dce775",
        ]
        if n <= len(base):
            return base[:n]
        extra = [f"hsl({int(360 * i / n)}, 70%, 65%)" for i in range(n - len(base))]
        return base + extra


class VisualizerRegistry:
    """Collects visualizer classes and dispatches to them by label."""

    def __init__(self, context: RenderContext) -> None:
        self._context = context
        self._factories: List[Callable[[RenderContext], BaseVisualizer]] = []

    def register(self, factory: Callable[[RenderContext], BaseVisualizer]) -> None:
        """Register a visualizer factory.

        Args:
            factory: Callable producing a :class:`BaseVisualizer` instance
                given the shared render context.
        """
        self._factories.append(factory)

    def build(self) -> List[BaseVisualizer]:
        """Instantiate all registered visualizers."""
        return [factory(self._context) for factory in self._factories]

    def applicable(self) -> List[BaseVisualizer]:
        """Return the subset of visualizers whose data is present."""
        return [v for v in self.build() if v.is_applicable()]


class SidebarController:
    """Renders the sidebar controls and returns user selections."""

    def __init__(self, config: ExplorerConfig) -> None:
        self._config = config

    def render(self) -> Dict[str, Any]:
        """Render the sidebar and return the current selections."""
        st.sidebar.markdown("## Checkpoint")
        uploaded = st.sidebar.file_uploader(
            "Upload checkpoint",
            type=[ext.lstrip(".") for ext in self._config.supported_checkpoint_extensions],
            help="safetensors, pt, bin or pth produced by topogpt2_1.py",
        )
        manual_path = st.sidebar.text_input(
            "...or path on disk",
            value="",
            help="Absolute or relative path to a checkpoint file on the server.",
        )
        st.sidebar.markdown("---")
        st.sidebar.markdown("## Rendering")
        st.sidebar.caption(
            "Sampling caps guarantee responsive rendering on large models."
        )
        st.sidebar.markdown(
            f"**Max rows for PCA:** {self._config.sampling.max_rows_for_pca}  \n"
            f"**Max cols for PCA:** {self._config.sampling.max_cols_for_pca}  \n"
            f"**Max heatmap elements:** {self._config.sampling.max_elements_heatmap}  \n"
            f"**Max FFT rows:** {self._config.sampling.max_fft_rows}  \n"
            f"**Histogram cap:** {self._config.sampling.max_elements_for_histogram}"
        )
        st.sidebar.markdown("---")
        st.sidebar.markdown("## References")
        st.sidebar.markdown(
            "- Parcollet et al. (2019): *Quaternion Recurrent Neural Networks*  \n"
            "- Li et al. (2021): *Fourier Neural Operator*  \n"
            "- Shazeer (2020): *GLU Variants Improve Transformer*  \n"
            "- Liu et al. (2022): *Grokking*  \n"
            "- Nanda et al. (2023): *Progress Measures via Mechanistic Interpretability*"
        )
        return {"uploaded": uploaded, "manual_path": manual_path.strip()}


class ExplorerApp:
    """Top-level orchestration: composes loader, registry, and layout."""

    def __init__(self, config: Optional[ExplorerConfig] = None) -> None:
        self._config = config or ExplorerConfig()
        self._loader = CheckpointLoader(self._config)
        self._classifier = TensorClassifier()
        self._projector = TensorProjector(self._config)
        self._metrics = MetricCalculator(self._config)
        self._styler = FigureStyler(self._config)
        self._sidebar = SidebarController(self._config)
        self._style_injector = StyleInjector(self._config.theme)

    def run(self) -> None:
        """Entry point used by ``streamlit run``."""
        self._configure_page()
        self._style_injector.inject()
        self._render_header()
        selections = self._sidebar.render()
        tensors = self._resolve_checkpoint(selections)
        if tensors is None:
            self._render_landing()
            return
        inventory = TensorInventory(tensors, self._classifier)
        context = RenderContext(
            inventory=inventory,
            projector=self._projector,
            metrics=self._metrics,
            styler=self._styler,
            config=self._config,
        )
        registry = self._build_registry(context)
        self._render_tabs(registry)
        self._render_footer()

    def _configure_page(self) -> None:
        st.set_page_config(
            page_title=self._config.page_title,
            layout="wide",
            initial_sidebar_state="expanded",
        )

    def _render_header(self) -> None:
        st.markdown('<div class="header-container">', unsafe_allow_html=True)
        st.title(self._config.page_title)
        st.subheader("Interactive 3D and 2D geometry of a TopoGPT2 checkpoint")
        st.markdown(
            '<div class="citation-box">'
            "Every visualization here is derived from the actual checkpoint tensors. "
            "No synthetic placeholders are introduced. Metrics (effective rank, "
            "participation ratio, spectral flatness, coherence, fractal dimension) "
            "are computed directly from the loaded parameters."
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    def _render_landing(self) -> None:
        st.markdown('<div class="header-container">', unsafe_allow_html=True)
        st.markdown(
            "### Upload a checkpoint to begin\n\n"
            "This explorer accepts TopoGPT2 checkpoints in `.safetensors`, `.pt`, "
            "`.bin`, or `.pth` format. Use the sidebar to upload a file or to point "
            "the explorer at a path on disk.\n\n"
            "Once a checkpoint is loaded, the tabs across the top expose views over:\n"
            "- the full parameter inventory (overview);\n"
            "- a single tensor's 3D point cloud, heatmap, distribution, spectrum, "
            "and singular value curve;\n"
            "- the (w, x, y, z) decomposition of quaternion layers;\n"
            "- complex spectral kernels in the frequency plane;\n"
            "- the learned torus topology graph and its edge quaternions;\n"
            "- attention Q/K/V/O per-head geometry;\n"
            "- MoE router behaviour and expert similarity;\n"
            "- how any role evolves across layers;\n"
            "- a global cross-tensor feature-space embedding."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    def _resolve_checkpoint(
        self, selections: Dict[str, Any]
    ) -> Optional[Dict[str, torch.Tensor]]:
        uploaded = selections.get("uploaded")
        manual_path = selections.get("manual_path")
        try:
            if uploaded is not None:
                return self._loader.load(uploaded)
            if manual_path:
                return self._loader.load(manual_path)
        except Exception as exc:  # surfaced to the user, no stack trace leakage
            st.error(f"Failed to load checkpoint: {exc}")
            return None
        return None

    def _build_registry(self, context: RenderContext) -> VisualizerRegistry:
        registry = VisualizerRegistry(context)
        registry.register(lambda c: OverviewVisualizer(c))
        registry.register(lambda c: TensorExplorerVisualizer(c))
        registry.register(lambda c: QuaternionDecompositionVisualizer(c))
        registry.register(lambda c: SpectralKernelVisualizer(c))
        registry.register(lambda c: TorusTopologyVisualizer(c))
        registry.register(lambda c: AttentionVisualizer(c))
        registry.register(lambda c: MoEVisualizer(c))
        registry.register(lambda c: LayerEvolutionVisualizer(c))
        registry.register(lambda c: GlobalGeometryVisualizer(c))
        return registry

    def _render_tabs(self, registry: VisualizerRegistry) -> None:
        visualizers = registry.applicable()
        if not visualizers:
            st.warning("No applicable visualizers for this checkpoint.")
            return
        labels = [v.label for v in visualizers]
        tabs = st.tabs(labels)
        for tab, viz in zip(tabs, visualizers):
            with tab:
                st.markdown('<div class="header-container">', unsafe_allow_html=True)
                st.markdown(f"#### {viz.label}")
                st.caption(viz.description)
                st.markdown("</div>", unsafe_allow_html=True)
                try:
                    viz.render()
                except Exception as exc:
                    st.error(f"Visualizer '{viz.label}' raised: {exc}")

    def _render_footer(self) -> None:
        st.markdown(
            '<div class="footer">'
            "<b>TopoGPT2 Weights Explorer</b> - "
            "A mechanistic-interpretability tool for quaternion-topological transformers."
            "</div>",
            unsafe_allow_html=True,
        )


def main() -> None:
    """Streamlit script entry point."""
    app = ExplorerApp()
    app.run()


if __name__ == "__main__":
    main()
