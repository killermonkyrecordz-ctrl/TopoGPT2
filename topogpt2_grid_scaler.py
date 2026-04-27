"""TopoGPT2 Zero-Shot D_MODEL Scaler.

Scales a trained TopoGPT2 checkpoint to a larger hidden dimension
(D_MODEL) without re-training, preserving the spectral structure of
every learned weight matrix.

This is the direct analogue of the Willmore Crystal zero-shot grid
scaler, where:

    Willmore Crystal          TopoGPT2
    ─────────────────         ──────────────────────────
    grid_size  16 → 128       D_MODEL  256 → 512 → 768
    SpectralLayer kernels     QuaternionLinear Ww/Wx/Wy/Wz
    channels  (fixed)         TORUS topology (fixed, untouched)

The torus topology — TORUS_RADIAL_BINS, TORUS_ANGULAR_BINS,
N_TORUS_NODES, edges_i / edges_j / edge_type, edge_quat — is
NEVER modified. It is the structural invariant of the model, exactly
as the number of channels is the invariant in the Willmore Crystal.

What changes with D_MODEL
─────────────────────────
  D_QUAT  = D_MODEL // 4         (quaternion sub-space size)
  D_HEAD  = D_MODEL // N_HEADS   (attention head size)
  SPECTRAL_LATENT_DIM            (AE bottleneck)
  n_freq  = D_MODEL // 2 + 1     (1D spectral filter length)

Tensor families and how each is scaled
───────────────────────────────────────
  quat_linear     [out_q, in_q]           2D Fourier interpolation
                                          on both axes (q = D//4)

  spectral_1d     [D//2+1]                1D Fourier: zero-pad or
                                          truncate frequency axis

  spectral_2d     [D_q, D_q, R, A//2+1]  Fourier interpolation on
                                          (in_q, out_q) axes; R and
                                          A//2+1 are TOPO-fixed and
                                          copied verbatim

  node_embed      [N_NODES, D]            Fourier interpolation of
                                          each node's feature vector
                                          along the D axis only

  vocab_proj      [VOCAB, D]              Fourier interpolation along D
  attn_proj       [*, D] or [D, *]        Fourier interpolation on D axes
  moe_router      [N_experts, D]          Fourier interpolation along D
  swiglu          [inner, D]              inner = f(D) → recomputed;
                                          interpolate to new inner dim
  norm / bias     [D] or scalar           Fourier interpolation (1D)
                                          or verbatim (scalar)

  topo_graph      edges_i/j/edge_type     VERBATIM — topology invariant
  edge_quat       [4, 4]                  VERBATIM — topology invariant
  rope            cos/sin/inv_freq        REBUILT for new D_HEAD

Scaling algorithm (mirrors Willmore Crystal)
────────────────────────────────────────────
For a weight matrix W of shape [M, N] scaled to [M', N']:
  1. FFT2 along (M, N).
  2. Zero-pad (upsample) or centre-crop (downsample) the spectrum
     to size (M', N').
  3. IFFT2 → W'.
  4. Rescale amplitude: ||W'||_F = ||W||_F.

For 1D vectors of length L → L':
  1. RFFT.
  2. Zero-pad / crop to L'//2+1 frequencies.
  3. IRFFT → v' of length L'.
  4. Rescale amplitude.

All thresholds, scale factors, and method choices live in
TopoScalerConfig. No magic numbers inline.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class InterpolationConfig:
    """Controls the spectral weight interpolation."""

    method: str = "fourier"
    amplitude_normalization: bool = True
    phase_preservation: bool = True
    amplitude_epsilon: float = 1e-10
    boundary_mode: str = "reflect"


@dataclass
class ValidationConfig:
    """What to measure after each scaling step."""

    compute_spectral_concentration: bool = True
    compute_phase_coherence: bool = True
    compute_norm_ratio: bool = True
    spectral_concentration_tolerance: float = 0.15
    phase_coherence_tolerance: float = 0.20
    norm_ratio_tolerance: float = 0.25
    abort_on_degradation: bool = False


@dataclass
class ProgressiveConfig:
    """Multi-hop scaling strategy."""

    enable: bool = True
    validate_each_step: bool = True
    save_intermediate: bool = True
    early_stop_on_degradation: bool = True


@dataclass
class OutputConfig:
    """Output files."""

    output_dir: str = "scaled_models"
    save_pytorch: bool = True
    save_json_metrics: bool = True
    save_report: bool = True


@dataclass
class TopoScalerConfig:
    """Top-level configuration — mirrors scaler_config_128.toml."""

    source_checkpoint: str = ""
    source_d_model: int = 256
    target_d_models: List[int] = field(default_factory=lambda: [512, 768])
    topogpt2_script: str = "topogpt2_1.py"
    device: str = "cpu"
    seed: int = 42
    log_level: str = "INFO"

    interpolation: InterpolationConfig = field(
        default_factory=InterpolationConfig
    )
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    progressive: ProgressiveConfig = field(default_factory=ProgressiveConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def __post_init__(self) -> None:
        for d in self.target_d_models:
            if d % 4 != 0:
                raise ValueError(
                    f"target D_MODEL={d} must be divisible by 4 (quaternion constraint)."
                )
            if d < 4:
                raise ValueError(f"target D_MODEL={d} must be >= 4.")


def _setup_logger(name: str, level: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        )
        logger.addHandler(h)
    return logger


class ModuleImporter:
    """Dynamically imports topogpt2_1.py from any path."""

    def load(self, path: str) -> Any:
        """Import and return the topogpt2 module."""
        import importlib.util

        p = Path(path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Model script not found: {p}")
        spec = importlib.util.spec_from_file_location("topogpt2_user", str(p))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create spec from {p}")
        mod = importlib.util.module_from_spec(spec)
        parent = str(p.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        spec.loader.exec_module(mod)
        for sym in ("TopoGPT2", "TopoGPT2Config"):
            if not hasattr(mod, sym):
                raise ImportError(f"Module at {p} is missing: {sym}")
        return mod


class CheckpointReader:
    """Reads a TopoGPT2 checkpoint and any embedded config."""

    def read(
        self, path: str, device: str
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, Any]]]:
        """Return (state_dict, optional_embedded_config)."""
        p = Path(path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        if p.suffix.lower() == ".safetensors":
            from safetensors.torch import load as st_load
            data = st_load(p.read_bytes())
            return {k: v.to(device) for k, v in data.items()}, None
        try:
            obj = torch.load(str(p), map_location=device, weights_only=True)
        except Exception:
            obj = torch.load(str(p), map_location=device, weights_only=False)
        return self._extract(obj)

    def _extract(
        self, obj: Any
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, Any]]]:
        embedded: Optional[Dict[str, Any]] = None
        if isinstance(obj, dict):
            if isinstance(obj.get("config"), dict):
                embedded = obj["config"]
            for key in ("model_state_dict", "state_dict", "model", "weights"):
                candidate = obj.get(key)
                if isinstance(candidate, dict) and candidate:
                    if isinstance(next(iter(candidate.values())), torch.Tensor):
                        return candidate, embedded
            if obj and isinstance(next(iter(obj.values())), torch.Tensor):
                return obj, embedded
        raise RuntimeError("No tensor state dict found in checkpoint.")


class ConfigReconstructor:
    """Reconstructs TopoGPT2Config from weights, inferring all dimensions."""

    def reconstruct(
        self,
        mod: Any,
        state_dict: Dict[str, torch.Tensor],
        embedded: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Return a TopoGPT2Config that matches the loaded weights exactly."""
        if embedded is not None:
            known = set(mod.TopoGPT2Config.__dataclass_fields__.keys())
            return mod.TopoGPT2Config(
                **{k: v for k, v in embedded.items() if k in known}
            )
        return self._infer(mod, state_dict)

    def _infer(self, mod: Any, sd: Dict[str, torch.Tensor]) -> Any:
        te = sd["token_embed.weight"]
        vocab, d_model = int(te.shape[0]), int(te.shape[1])
        n_layers = (
            max(
                int(m.group(1))
                for k in sd
                for m in [re.match(r"layers\.(\d+)\.", k)]
                if m
            )
            + 1
        )
        d_head = self._infer_d_head(sd)
        n_heads = d_model // d_head if d_head else 8
        n_kv = self._infer_n_kv(sd, d_head, n_heads)
        max_seq = self._infer_max_seq(sd)
        radial, angular = self._infer_torus(sd)
        kwargs: Dict[str, Any] = dict(
            SCALE="custom",
            D_MODEL=d_model,
            VOCAB_SIZE=vocab,
            N_HEADS=n_heads,
            N_LAYERS=n_layers,
            N_KV_HEADS=n_kv,
            TORUS_RADIAL_BINS=radial,
            TORUS_ANGULAR_BINS=angular,
        )
        if max_seq is not None:
            kwargs["MAX_SEQ_LEN"] = max_seq
        return mod.TopoGPT2Config(**kwargs)

    @staticmethod
    def _infer_d_head(sd: Dict[str, torch.Tensor]) -> int:
        for key, t in sd.items():
            if key.endswith("rope.cos_cache") and t.dim() >= 2:
                return int(t.shape[1])
        return 32

    @staticmethod
    def _infer_n_kv(
        sd: Dict[str, torch.Tensor], d_head: int, n_heads: int
    ) -> int:
        kw = sd.get("layers.0.attn.k_proj.weight")
        if kw is None or d_head <= 0:
            return n_heads
        kv = int(kw.shape[0]) // d_head
        return kv if kv > 0 and n_heads % kv == 0 else n_heads

    @staticmethod
    def _infer_max_seq(sd: Dict[str, torch.Tensor]) -> Optional[int]:
        for key, t in sd.items():
            if key.endswith("rope.cos_cache") and t.dim() >= 1:
                return int(t.shape[0])
        return None

    @staticmethod
    def _infer_torus(sd: Dict[str, torch.Tensor]) -> Tuple[int, int]:
        for key, t in sd.items():
            if "torus_spectral.0.kr_w" in key and t.dim() == 4:
                return int(t.shape[2]), (int(t.shape[3]) - 1) * 2
        ne = sd.get("layers.0.topo_brain.shared_expert.node_embed")
        if ne is not None and ne.dim() == 2:
            n = int(ne.shape[0])
            for r in range(2, int(math.isqrt(n)) + 1):
                if n % r == 0:
                    return r, n // r
        return 2, 4


class TensorRole:
    """Classifies every state-dict key by its D_MODEL scaling role."""

    _TOPO_GRAPH = re.compile(r"\.(edges_i|edges_j|edge_type)$")
    _EDGE_QUAT = re.compile(r"\.edge_quat$")
    _ROPE = re.compile(r"rope\.(cos_cache|sin_cache|inv_freq)$")
    _SPECTRAL_2D = re.compile(r"torus_spectral\.\d+\.(kr|ki)_[wxyz]$")
    _SPECTRAL_1D = re.compile(r"\.(enc_kr|enc_ki|dec_kr|dec_ki)$")
    _NODE_EMBED = re.compile(r"\.node_embed$")
    _QUAT_LIN_W = re.compile(r"\.(Ww|Wx|Wy|Wz)\.weight$")
    _QUAT_LIN_B = re.compile(r"\.bias$")

    def classify(self, key: str, shape: Tuple[int, ...]) -> str:
        """Return a semantic role string.

        Roles:
          topo_graph    — graph index buffers (topology invariant, verbatim)
          edge_quat     — learned edge quaternions (topology invariant, verbatim)
          rope          — RoPE cache (rebuilt for new D_HEAD)
          spectral_2d   — 2D quaternion spectral kernels [D_q, D_q, R, Af]
          spectral_1d   — 1D spectral AE filters [D//2+1]
          node_embed    — torus node embeddings [N_NODES, D]
          quat_linear   — QuaternionLinear component weights [out_q, in_q]
          matrix        — any other 2D weight [M, N] where both dims scale with D
          vector        — any 1D parameter [D] (norms, biases)
          scalar        — 0D parameter (temperature)
          verbatim      — anything not classified above
        """
        if self._TOPO_GRAPH.search(key):
            return "topo_graph"
        if self._EDGE_QUAT.search(key):
            return "edge_quat"
        if self._ROPE.search(key):
            return "rope"
        if self._SPECTRAL_2D.search(key):
            return "spectral_2d"
        if self._SPECTRAL_1D.search(key):
            return "spectral_1d"
        if self._NODE_EMBED.search(key):
            return "node_embed"
        if self._QUAT_LIN_W.search(key):
            return "quat_linear"
        ndim = len(shape)
        if ndim == 0:
            return "scalar"
        if ndim == 1:
            return "vector"
        if ndim == 2:
            return "matrix"
        return "verbatim"


class SpectralInterpolator:
    """Interpolates weight tensors via Fourier-space zero-padding or cropping.

    This is the core of the technique — identical in principle to the
    Willmore Crystal scaler's Fourier interpolation of spectral kernels,
    adapted to weight matrices of arbitrary shape.
    """

    def __init__(self, cfg: InterpolationConfig) -> None:
        self._cfg = cfg

    def interpolate_2d(
        self,
        W: torch.Tensor,
        tgt_rows: int,
        tgt_cols: int,
    ) -> torch.Tensor:
        """Scale a 2D weight matrix [M, N] to [M', N'] via spectral interpolation.

        Steps:
          1. FFT2 of W.
          2. Zero-pad or centre-crop frequency spectrum to (M', N').
          3. IFFT2.
          4. Amplitude normalisation: ||W'||_F = ||W||_F.
        """
        src_norm = W.float().norm()
        M, N = W.shape

        W_f = torch.fft.fft2(W.float())

        W_f_scaled = self._resize_spectrum_2d(W_f, tgt_rows, tgt_cols)

        W_prime = torch.fft.ifft2(W_f_scaled).real

        if self._cfg.amplitude_normalization:
            tgt_norm = W_prime.norm()
            if tgt_norm > self._cfg.amplitude_epsilon:
                W_prime = W_prime * (src_norm / tgt_norm)

        return W_prime.to(W.dtype)

    def interpolate_1d(
        self,
        v: torch.Tensor,
        tgt_len: int,
    ) -> torch.Tensor:
        """Scale a 1D vector of length L to length L' via spectral interpolation."""
        src_norm = v.float().norm()
        L = v.shape[0]

        V_f = torch.fft.rfft(v.float())

        src_freqs = L // 2 + 1
        tgt_freqs = tgt_len // 2 + 1

        if tgt_freqs >= src_freqs:
            pad = tgt_freqs - src_freqs
            V_f_scaled = F.pad(V_f, (0, pad))
        else:
            V_f_scaled = V_f[:tgt_freqs]

        V_f_scaled = V_f_scaled * (tgt_len / max(L, 1)) ** 0.5

        v_prime = torch.fft.irfft(V_f_scaled, n=tgt_len)

        if self._cfg.amplitude_normalization:
            p_norm = v_prime.norm()
            if p_norm > self._cfg.amplitude_epsilon:
                v_prime = v_prime * (src_norm / p_norm)

        return v_prime.to(v.dtype)

    def _resize_spectrum_2d(
        self,
        W_f: torch.Tensor,
        tgt_rows: int,
        tgt_cols: int,
    ) -> torch.Tensor:
        """Zero-pad or centre-crop a 2D complex spectrum."""
        M, N = W_f.shape

        W_f_shift = torch.fft.fftshift(W_f)

        out = torch.zeros(tgt_rows, tgt_cols, dtype=W_f.dtype, device=W_f.device)

        copy_r = min(M, tgt_rows)
        copy_c = min(N, tgt_cols)

        src_r0 = (M - copy_r) // 2
        src_c0 = (N - copy_c) // 2
        tgt_r0 = (tgt_rows - copy_r) // 2
        tgt_c0 = (tgt_cols - copy_c) // 2

        out[tgt_r0: tgt_r0 + copy_r, tgt_c0: tgt_c0 + copy_c] = (
            W_f_shift[src_r0: src_r0 + copy_r, src_c0: src_c0 + copy_c]
        )

        scale = (tgt_rows * tgt_cols) / max(M * N, 1)
        out = torch.fft.ifftshift(out) * scale

        return out


class StateScaler:
    """Scales every tensor in a state dict from src_d to tgt_d.

    Routing table:
      topo_graph   → verbatim copy
      edge_quat    → verbatim copy
      rope         → rebuilt from scratch for new D_HEAD
      spectral_2d  → interpolate channel dims (D_q axes); topo dims verbatim
      spectral_1d  → interpolate 1D frequency axis (D//2+1 → D'//2+1)
      node_embed   → interpolate feature axis (D → D')
      quat_linear  → 2D spectral interpolation [out_q, in_q] → [out_q', in_q']
      matrix       → 2D spectral interpolation on D-dependent axes
      vector       → 1D spectral interpolation
      scalar       → verbatim copy
    """

    def __init__(
        self,
        interp: SpectralInterpolator,
        role_clf: TensorRole,
        logger: logging.Logger,
    ) -> None:
        self._interp = interp
        self._clf = role_clf
        self._log = logger

    def scale(
        self,
        src_state: Dict[str, torch.Tensor],
        src_cfg: Any,
        tgt_cfg: Any,
        mod: Any,
    ) -> Dict[str, torch.Tensor]:
        """Produce a complete scaled state dict."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tgt_model = mod.TopoGPT2(tgt_cfg)

        tgt_state = tgt_model.state_dict()
        rope_keys = {
            k for k in src_state
            if self._clf.classify(k, tuple(src_state[k].shape)) == "rope"
        }
        topo_keys = {
            k for k in src_state
            if self._clf.classify(k, tuple(src_state[k].shape)) == "topo_graph"
        }

        scaled: Dict[str, torch.Tensor] = {}
        for key, tensor in src_state.items():
            role = self._clf.classify(key, tuple(tensor.shape))

            if role == "topo_graph":
                scaled[key] = tgt_state.get(key, tensor).clone()
                continue

            if role == "edge_quat":
                scaled[key] = tensor.clone()
                continue

            if role == "rope":
                scaled[key] = tgt_state.get(key, tensor).clone()
                continue

            tgt_shape = tgt_state[key].shape if key in tgt_state else None
            if tgt_shape is None:
                scaled[key] = tensor.clone()
                self._log.debug("not in target model, copying verbatim: %s", key)
                continue

            if tuple(tgt_shape) == tuple(tensor.shape):
                scaled[key] = tensor.clone()
                continue

            scaled[key] = self._dispatch(tensor, tgt_shape, role, key)
            self._log.debug(
                "%s %s  %s -> %s",
                role, key, tuple(tensor.shape), tuple(tgt_shape),
            )

        return scaled

    def _dispatch(
        self,
        src: torch.Tensor,
        tgt_shape: Tuple[int, ...],
        role: str,
        key: str,
    ) -> torch.Tensor:
        """Route to the correct interpolation method based on shape and role."""
        ndim_src = src.dim()
        ndim_tgt = len(tgt_shape)

        if ndim_src == 0 or ndim_tgt == 0:
            return src.clone()

        if ndim_src == 1 and ndim_tgt == 1:
            return self._interp.interpolate_1d(src, int(tgt_shape[0]))

        if ndim_src == 2 and ndim_tgt == 2:
            return self._interp.interpolate_2d(
                src, int(tgt_shape[0]), int(tgt_shape[1])
            )

        if ndim_src == 4 and ndim_tgt == 4 and role == "spectral_2d":
            return self._scale_spectral_2d_to(src, tgt_shape, key)

        self._log.warning(
            "Cannot scale %s from %s to %s; using bilinear fallback.",
            key, tuple(src.shape), tuple(tgt_shape),
        )
        return self._bilinear_fallback(src, tgt_shape)

    def _scale_spectral_2d_to(
        self,
        t: torch.Tensor,
        tgt_shape: Tuple[int, ...],
        key: str,
    ) -> torch.Tensor:
        """Scale [in_q, out_q, R, Af] to [in_q', out_q', R, Af].

        R and Af are topo-invariant and must not change.
        """
        in_q, out_q, R, Af = t.shape
        tgt_inq, tgt_outq, tgt_R, tgt_Af = tgt_shape

        if R != tgt_R or Af != tgt_Af:
            self._log.warning(
                "spectral_2d %s: topo dims changed unexpectedly "
                "(%d,%d) -> (%d,%d); this should not happen.",
                key, R, Af, tgt_R, tgt_Af,
            )

        result = torch.zeros(tgt_inq, tgt_outq, R, Af, dtype=t.dtype)
        for ri in range(R):
            for ai in range(Af):
                result[:, :, ri, ai] = self._interp.interpolate_2d(
                    t[:, :, ri, ai].float(), tgt_inq, tgt_outq
                )
        return result

    def _bilinear_fallback(
        self, src: torch.Tensor, tgt_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        flat = src.float().reshape(1, 1, -1, 1)
        tgt_n = int(math.prod(tgt_shape))
        scaled = F.interpolate(flat, size=(tgt_n, 1), mode="bilinear", align_corners=False)
        return scaled.reshape(tgt_shape).to(src.dtype)


class ScalingValidator:
    """Measures spectral structure preservation before and after scaling."""

    def __init__(self, cfg: ValidationConfig) -> None:
        self._cfg = cfg

    def compute(
        self, state: Dict[str, torch.Tensor], d_model: int
    ) -> Dict[str, float]:
        """Compute all configured validation metrics."""
        out: Dict[str, float] = {"d_model": float(d_model)}
        out["param_count"] = float(sum(t.numel() for t in state.values()))
        out["total_frobenius"] = float(
            sum(t.float().norm().item() ** 2 for t in state.values()) ** 0.5
        )
        if self._cfg.compute_spectral_concentration:
            out["spectral_concentration"] = self._spectral_concentration(state)
        if self._cfg.compute_phase_coherence:
            out["phase_coherence"] = self._phase_coherence(state)
        return out

    def check_degradation(
        self,
        before: Dict[str, float],
        after: Dict[str, float],
        logger: logging.Logger,
    ) -> bool:
        """Return True if any metric dropped beyond its tolerance."""
        degraded = False
        checks = [
            ("spectral_concentration", self._cfg.spectral_concentration_tolerance),
            ("phase_coherence", self._cfg.phase_coherence_tolerance),
        ]
        for metric, tol in checks:
            if metric not in before or metric not in after:
                continue
            b, a = before[metric], after[metric]
            if b < 1e-12:
                continue
            drop = (b - a) / b
            if drop > tol:
                logger.warning(
                    "%s dropped %.1f%% (%.4f -> %.4f), tolerance %.0f%%.",
                    metric, drop * 100, b, a, tol * 100,
                )
                degraded = True
        if "total_frobenius" in before and "total_frobenius" in after:
            ratio = after["total_frobenius"] / max(before["total_frobenius"], 1e-12)
            tol = self._cfg.norm_ratio_tolerance
            if abs(ratio - 1.0) > tol:
                logger.warning(
                    "Frobenius ratio = %.4f (tolerance ±%.2f).", ratio, tol
                )
        return degraded

    def _spectral_concentration(self, sd: Dict[str, torch.Tensor]) -> float:
        scores = []
        clf = TensorRole()
        for key, t in sd.items():
            if clf.classify(key, tuple(t.shape)) != "quat_linear":
                continue
            if t.dim() != 2 or min(t.shape) < 2:
                continue
            W_f = torch.fft.fft2(t.float())
            mag = W_f.abs()
            total = mag.sum().item()
            if total < 1e-12:
                continue
            k = min(4, mag.numel())
            top = torch.topk(mag.reshape(-1), k).values.sum().item()
            scores.append(top / total)
        return float(np.mean(scores)) if scores else 0.0

    def _phase_coherence(self, sd: Dict[str, torch.Tensor]) -> float:
        scores = []
        clf = TensorRole()
        for key, t in sd.items():
            if clf.classify(key, tuple(t.shape)) != "quat_linear":
                continue
            if t.dim() != 2 or min(t.shape) < 2:
                continue
            W_f = torch.fft.fft2(t.float())
            phase = W_f.angle()
            cos_var = torch.cos(phase).var().item()
            scores.append(1.0 - min(cos_var, 1.0))
        return float(np.mean(scores)) if scores else 0.0


class ModelAssembler:
    """Loads scaled weights into a fresh TopoGPT2."""

    def assemble(
        self,
        mod: Any,
        tgt_cfg: Any,
        scaled_state: Dict[str, torch.Tensor],
        logger: logging.Logger,
    ) -> Any:
        """Instantiate the target model and load scaled weights."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = mod.TopoGPT2(tgt_cfg)

        missing, unexpected = model.load_state_dict(scaled_state, strict=False)
        if unexpected:
            logger.warning("Unexpected keys: %s", unexpected[:6])
        if missing:
            logger.warning(
                "%d missing keys (may be expected): %s", len(missing), missing[:6]
            )
        model.eval()
        return model


class CheckpointSaver:
    """Saves checkpoint, metrics JSON, and text report."""

    def save(
        self,
        model: Any,
        tgt_cfg: Any,
        src_cfg: Any,
        metrics_before: Dict[str, float],
        metrics_after: Dict[str, float],
        out_cfg: OutputConfig,
        step_tag: str,
        logger: logging.Logger,
    ) -> str:
        """Persist scaled checkpoint and metadata. Returns checkpoint path."""
        os.makedirs(out_cfg.output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"topogpt2_scaled_{step_tag}_{ts}"

        def _cfg_dict(c: Any) -> Dict[str, Any]:
            return {
                k: v
                for k, v in vars(c).items()
                if isinstance(v, (int, float, str, bool))
            }

        payload = {
            "model_state_dict": model.state_dict(),
            "config": _cfg_dict(tgt_cfg),
            "source_config": _cfg_dict(src_cfg),
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "scaled_at": ts,
            "scaler": "topogpt2_grid_scaler v2.0",
        }

        ckpt_path = os.path.join(out_cfg.output_dir, f"{base}.pt")
        if out_cfg.save_pytorch:
            torch.save(payload, ckpt_path)
            logger.info("Saved: %s", ckpt_path)

        if out_cfg.save_json_metrics:
            jpath = os.path.join(out_cfg.output_dir, f"{base}_metrics.json")
            with open(jpath, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "metrics_before": metrics_before,
                        "metrics_after": metrics_after,
                        "config": payload["config"],
                        "source_config": payload["source_config"],
                    },
                    f, indent=2,
                )
            logger.info("Metrics: %s", jpath)

        if out_cfg.save_report:
            rpath = os.path.join(out_cfg.output_dir, f"{base}_report.txt")
            self._write_report(
                rpath, src_cfg, tgt_cfg, metrics_before, metrics_after, ckpt_path
            )
            logger.info("Report: %s", rpath)

        return ckpt_path

    def _write_report(
        self,
        path: str,
        src_cfg: Any,
        tgt_cfg: Any,
        before: Dict[str, float],
        after: Dict[str, float],
        ckpt: str,
    ) -> None:
        lines = [
            "=" * 72,
            "TopoGPT2 Zero-Shot D_MODEL Scaler — Report",
            "=" * 72,
            "",
            f"Checkpoint  : {ckpt}",
            f"Timestamp   : {datetime.now().isoformat()}",
            "",
            f"D_MODEL     : {int(src_cfg.D_MODEL)} -> {int(tgt_cfg.D_MODEL)}",
            f"D_QUAT      : {int(src_cfg.D_QUAT)} -> {int(tgt_cfg.D_QUAT)}",
            f"TORUS       : {src_cfg.TORUS_RADIAL_BINS}x{src_cfg.TORUS_ANGULAR_BINS} (unchanged)",
            f"N_TORUS_NODES: {src_cfg.N_TORUS_NODES} (unchanged)",
            f"N_HEADS     : {src_cfg.N_HEADS} -> {tgt_cfg.N_HEADS}",
            f"N_LAYERS    : {src_cfg.N_LAYERS} (unchanged)",
            f"VOCAB_SIZE  : {src_cfg.VOCAB_SIZE} (unchanged)",
            "",
            "Validation metrics:",
        ]
        for k in sorted(set(before) | set(after)):
            b = before.get(k, float("nan"))
            a = after.get(k, float("nan"))
            if abs(b) > 1e-12:
                pct = (a - b) / abs(b) * 100
                sign = "+" if pct >= 0 else ""
                lines.append(
                    f"  {k:<35} {b:>12.6f}  ->  {a:>12.6f}  ({sign}{pct:.1f}%)"
                )
            else:
                lines.append(
                    f"  {k:<35} {b:>12.6f}  ->  {a:>12.6f}"
                )
        lines += ["", "=" * 72]
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


class TopoGPT2DModelScaler:
    """Main orchestrator.

    Mirrors the Willmore Crystal scaler pipeline:

      source_grid_size = D_MODEL_src
      target_grid_sizes = [D1, D2, ...]  (progressive)

    The torus topology is the structural invariant: RADIAL, ANGULAR,
    N_TORUS_NODES, edges_i/j/type, edge_quat are never modified.
    """

    def __init__(self, cfg: TopoScalerConfig) -> None:
        self._cfg = cfg
        self._log = _setup_logger("TopoGPT2Scaler", cfg.log_level)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        self._importer = ModuleImporter()
        self._reader = CheckpointReader()
        self._reconstructor = ConfigReconstructor()
        self._clf = TensorRole()
        self._interp = SpectralInterpolator(cfg.interpolation)
        self._state_scaler = StateScaler(self._interp, self._clf, self._log)
        self._validator = ScalingValidator(cfg.validation)
        self._assembler = ModelAssembler()
        self._saver = CheckpointSaver()

    def run(self) -> List[Dict[str, Any]]:
        """Execute the full scaling pipeline. Returns per-step result dicts."""
        cfg = self._cfg
        self._log.info("=" * 72)
        self._log.info("TopoGPT2 Zero-Shot D_MODEL Scaler")
        self._log.info("Source D_MODEL=%d -> targets=%s", cfg.source_d_model, cfg.target_d_models)
        self._log.info("=" * 72)

        mod = self._importer.load(cfg.topogpt2_script)
        state, embedded = self._reader.read(cfg.source_checkpoint, cfg.device)
        src_cfg = self._reconstructor.reconstruct(mod, state, embedded)

        self._log.info(
            "Source: D_MODEL=%d  D_QUAT=%d  TORUS=%dx%d  N_HEADS=%d  N_LAYERS=%d  VOCAB=%d",
            src_cfg.D_MODEL, src_cfg.D_QUAT,
            src_cfg.TORUS_RADIAL_BINS, src_cfg.TORUS_ANGULAR_BINS,
            src_cfg.N_HEADS, src_cfg.N_LAYERS, src_cfg.VOCAB_SIZE,
        )
        self._log.info(
            "  params: %d  |  torus nodes: %d (topology invariant)",
            sum(t.numel() for t in state.values()),
            src_cfg.N_TORUS_NODES,
        )

        metrics_before = self._validator.compute(state, src_cfg.D_MODEL)
        self._log.info(
            "Source metrics: spectral_conc=%.4f  phase_coh=%.4f  frobenius=%.2f",
            metrics_before.get("spectral_concentration", 0),
            metrics_before.get("phase_coherence", 0),
            metrics_before.get("total_frobenius", 0),
        )

        current_state = state
        current_cfg = src_cfg
        results: List[Dict[str, Any]] = []

        for step_idx, tgt_d in enumerate(cfg.target_d_models):
            self._log.info("-" * 72)
            self._log.info(
                "Step %d/%d: D_MODEL %d -> %d",
                step_idx + 1, len(cfg.target_d_models),
                current_cfg.D_MODEL, tgt_d,
            )

            tgt_cfg_model = self._build_target_config(mod, current_cfg, tgt_d)
            self._log.info(
                "Target: D_MODEL=%d  D_QUAT=%d  D_HEAD=%d  SPECTRAL_LATENT=%d  N_HEADS=%d",
                tgt_cfg_model.D_MODEL, tgt_cfg_model.D_QUAT,
                tgt_cfg_model.D_HEAD, tgt_cfg_model.SPECTRAL_LATENT_DIM,
                tgt_cfg_model.N_HEADS,
            )

            t0 = time.perf_counter()
            scaled_state = self._state_scaler.scale(
                current_state, current_cfg, tgt_cfg_model, mod
            )
            dt = time.perf_counter() - t0
            self._log.info("Scaling completed in %.2f s", dt)

            model = self._assembler.assemble(
                mod, tgt_cfg_model, scaled_state, self._log
            )
            n_params = sum(p.numel() for p in model.parameters())
            self._log.info("Assembled: %d parameters", n_params)

            metrics_after = self._validator.compute(scaled_state, tgt_d)
            self._log.info(
                "Scaled metrics: spectral_conc=%.4f  phase_coh=%.4f  frobenius=%.2f",
                metrics_after.get("spectral_concentration", 0),
                metrics_after.get("phase_coherence", 0),
                metrics_after.get("total_frobenius", 0),
            )

            degraded = self._validator.check_degradation(
                metrics_before, metrics_after, self._log
            )

            step_tag = f"D{tgt_d}"
            ckpt_path = self._saver.save(
                model, tgt_cfg_model, current_cfg,
                metrics_before, metrics_after,
                cfg.output, step_tag, self._log,
            )

            result: Dict[str, Any] = {
                "step": step_idx + 1,
                "src_d_model": int(current_cfg.D_MODEL),
                "tgt_d_model": tgt_d,
                "src_params": int(sum(t.numel() for t in current_state.values())),
                "tgt_params": n_params,
                "torus_radial": int(src_cfg.TORUS_RADIAL_BINS),
                "torus_angular": int(src_cfg.TORUS_ANGULAR_BINS),
                "n_torus_nodes": int(src_cfg.N_TORUS_NODES),
                "degraded": degraded,
                "metrics_before": metrics_before,
                "metrics_after": metrics_after,
                "checkpoint": ckpt_path,
                "elapsed_s": dt,
            }
            results.append(result)

            if degraded and cfg.progressive.early_stop_on_degradation and cfg.validation.abort_on_degradation:
                self._log.error("Aborting: quality degradation at step %d.", step_idx + 1)
                break

            if cfg.progressive.enable:
                current_state = scaled_state
                current_cfg = tgt_cfg_model
                metrics_before = metrics_after

        self._print_summary(results, src_cfg)
        return results

    def _build_target_config(self, mod: Any, src_cfg: Any, tgt_d: int) -> Any:
        """Construct a target config with new D_MODEL, preserving torus topology."""
        known = set(mod.TopoGPT2Config.__dataclass_fields__.keys())
        d = {k: v for k, v in vars(src_cfg).items() if k in known}
        d["SCALE"] = "custom"
        d["D_MODEL"] = tgt_d
        for derived in ("D_QUAT", "D_HEAD", "SPECTRAL_LATENT_DIM",
                        "N_TORUS_NODES", "GQA_GROUPS"):
            d.pop(derived, None)
        tgt_n_heads = self._infer_n_heads(tgt_d, src_cfg.N_HEADS)
        d["N_HEADS"] = tgt_n_heads
        d["N_KV_HEADS"] = self._infer_n_kv_heads(
            tgt_d, tgt_n_heads, src_cfg.N_HEADS, src_cfg.N_KV_HEADS
        )
        return mod.TopoGPT2Config(**d)

    @staticmethod
    def _infer_n_heads(tgt_d: int, src_n_heads: int) -> int:
        """Find the largest divisor of tgt_d that keeps D_HEAD >= 8."""
        if tgt_d % src_n_heads == 0:
            d_head = tgt_d // src_n_heads
            if d_head >= 8 and d_head % 2 == 0:
                return src_n_heads
        for n in (src_n_heads * 2, src_n_heads, src_n_heads // 2,
                  16, 12, 8, 6, 4, 2, 1):
            if n > 0 and tgt_d % n == 0:
                d_head = tgt_d // n
                if d_head >= 8 and d_head % 2 == 0:
                    return n
        return 1

    @staticmethod
    def _infer_n_kv_heads(
        tgt_d: int,
        tgt_n_heads: int,
        src_n_heads: int,
        src_n_kv: int,
    ) -> int:
        ratio = src_n_kv / max(src_n_heads, 1)
        tgt_kv = max(1, round(tgt_n_heads * ratio))
        while tgt_n_heads % tgt_kv != 0:
            tgt_kv -= 1
            if tgt_kv <= 0:
                return tgt_n_heads
        return tgt_kv

    def _print_summary(
        self, results: List[Dict[str, Any]], src_cfg: Any
    ) -> None:
        self._log.info("=" * 72)
        self._log.info("SCALING COMPLETE — %d step(s)", len(results))
        self._log.info(
            "  Torus topology preserved: %dx%d = %d nodes",
            src_cfg.TORUS_RADIAL_BINS,
            src_cfg.TORUS_ANGULAR_BINS,
            src_cfg.N_TORUS_NODES,
        )
        for r in results:
            status = "DEGRADED" if r["degraded"] else "OK"
            ratio = r["tgt_params"] / max(r["src_params"], 1)
            self._log.info(
                "  Step %d: D_MODEL %d -> %d  params %d->%d (×%.2f)  [%s]",
                r["step"], r["src_d_model"], r["tgt_d_model"],
                r["src_params"], r["tgt_params"], ratio, status,
            )
            self._log.info("    -> %s", r["checkpoint"])
        self._log.info("=" * 72)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="TopoGPT2 Zero-Shot D_MODEL Scaler",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True,
                   help="Source checkpoint (.pt/.pth/.safetensors)")
    p.add_argument("--script", default="topogpt2_1.py",
                   help="Path to topogpt2_1.py")
    p.add_argument("--source-d-model", type=int, default=256,
                   help="Source D_MODEL (auto-inferred if checkpoint has embedded config)")
    p.add_argument("--target-d-models", nargs="+", type=int,
                   default=[512, 768],
                   help="Target D_MODEL values (progressive scaling)")
    p.add_argument("--output-dir", default="scaled_models")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-progressive", action="store_true",
                   help="Scale each target independently from source")
    p.add_argument("--no-amplitude-norm", action="store_true",
                   help="Skip amplitude normalisation")
    p.add_argument("--abort-on-degradation", action="store_true")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING"])
    return p


def config_from_args(args: argparse.Namespace) -> TopoScalerConfig:
    return TopoScalerConfig(
        source_checkpoint=args.checkpoint,
        source_d_model=args.source_d_model,
        target_d_models=args.target_d_models,
        topogpt2_script=args.script,
        device=args.device,
        seed=args.seed,
        log_level=args.log_level,
        interpolation=InterpolationConfig(
            amplitude_normalization=not args.no_amplitude_norm,
        ),
        validation=ValidationConfig(
            abort_on_degradation=args.abort_on_degradation,
        ),
        progressive=ProgressiveConfig(
            enable=not args.no_progressive,
        ),
        output=OutputConfig(output_dir=args.output_dir),
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    cfg = config_from_args(args)
    scaler = TopoGPT2DModelScaler(cfg)
    try:
        results = scaler.run()
        failed = [r for r in results if r["degraded"]]
        return 1 if failed and cfg.validation.abort_on_degradation else 0
    except FileNotFoundError as e:
        logging.getLogger("TopoGPT2Scaler").error("File not found: %s", e)
        return 2
    except Exception as e:
        logging.getLogger("TopoGPT2Scaler").error("Error: %s", e, exc_info=True)
        return 3


if __name__ == "__main__":
    sys.exit(main())
