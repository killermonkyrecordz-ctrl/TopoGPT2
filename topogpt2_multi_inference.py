#!/usr/bin/env python3
"""
topogpt2_multi_inference.py

Runs a single prompt through every TopoGPT2 checkpoint found in a
directory (or an explicit list) and prints the outputs side by side.

Supports:
  - Scaled checkpoints produced by topogpt2_dmodel_scaler.py (.pt with
    embedded config in 'config' key)
  - Original training checkpoints saved as .safetensors or plain .pt
    state dicts

The architecture config is always inferred directly from the checkpoint
weights (token_embed shape, k_proj shape, RoPE cache, torus kernels) so
no --scale flag is needed.  Every checkpoint is loaded, run, and unloaded
sequentially to keep peak RAM bounded.

Usage
-----
  python topogpt2_multi_inference.py \\
      --prompt "Once upon a time" \\
      --checkpoints scaled_models/ \\
      --script topogpt2_1.py \\
      --max-new 200 \\
      --temp 0.8 \\
      --top-k 40

  # Or pass explicit files:
  python topogpt2_multi_inference.py \\
      --prompt "The quantum torus" \\
      --checkpoints \\
          checkpoints_topogpt2/best/model.safetensors \\
          scaled_models/topogpt2_scaled_D512_*.pt \\
          scaled_models/topogpt2_scaled_D768_*.pt \\
      --script topogpt2_1.py
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SamplingConfig:
    """Generation hyper-parameters."""

    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    seed: int = 42


@dataclass(frozen=True)
class RunConfig:
    """Top-level execution configuration."""

    prompt: str = "Once upon a time"
    checkpoints: List[str] = field(default_factory=list)
    script: str = "topogpt2_1.py"
    device: str = "cpu"
    log_level: str = "INFO"
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    side_by_side: bool = True
    save_json: str = ""


def _setup_logger(name: str, level: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(h)
    return logger


class ModuleImporter:
    """Imports topogpt2_1.py from any filesystem path (cached per process)."""

    _cache: Dict[str, Any] = {}

    def load(self, path: str) -> Any:
        """Return the imported module, reusing the cached copy if available."""
        resolved = str(Path(path).expanduser().resolve())
        if resolved in self._cache:
            return self._cache[resolved]
        if not Path(resolved).exists():
            raise FileNotFoundError(f"Model script not found: {resolved}")
        spec = importlib.util.spec_from_file_location("topogpt2_user", resolved)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load spec from {resolved}")
        mod = importlib.util.module_from_spec(spec)
        parent = str(Path(resolved).parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        spec.loader.exec_module(mod)
        for sym in ("TopoGPT2", "TopoGPT2Config", "BPETokenizer"):
            if not hasattr(mod, sym):
                raise ImportError(f"Module at {resolved} is missing: {sym}")
        self._cache[resolved] = mod
        return mod


class CheckpointDiscovery:
    """Resolves a mixed list of files and directories to concrete checkpoint paths."""

    SUPPORTED_EXTENSIONS = {".pt", ".pth", ".safetensors", ".bin"}

    def resolve(self, sources: List[str]) -> List[Path]:
        """Expand directories and glob patterns into a sorted list of paths.

        Args:
            sources: File paths, directory paths, or glob patterns.

        Returns:
            Deduplicated, sorted list of existing checkpoint paths.
        """
        found: List[Path] = []
        seen: set = set()
        for source in sources:
            p = Path(source)
            if p.is_dir():
                for ext in self.SUPPORTED_EXTENSIONS:
                    for candidate in sorted(p.rglob(f"*{ext}")):
                        if candidate.is_file() and candidate not in seen:
                            found.append(candidate)
                            seen.add(candidate)
            elif p.is_file():
                if p not in seen:
                    found.append(p)
                    seen.add(p)
            else:
                import glob
                for match in sorted(glob.glob(str(source))):
                    mp = Path(match)
                    if mp.is_file() and mp not in seen:
                        found.append(mp)
                        seen.add(mp)
        if not found:
            raise FileNotFoundError(
                f"No checkpoint files found in: {sources}"
            )
        return found


class CheckpointLoader:
    """Loads state dicts and any embedded config from checkpoint files."""

    def load(
        self, path: Path, device: str
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, Any]]]:
        """Return (state_dict, optional_embedded_config).

        Handles:
          - .safetensors (raw state dict)
          - .pt / .pth with {'model_state_dict': ..., 'config': ...}
            (produced by the scaler)
          - .pt / .pth plain state dicts
        """
        suffix = path.suffix.lower()
        if suffix == ".safetensors":
            return self._load_safetensors(path, device)
        return self._load_torch(path, device)

    def _load_safetensors(
        self, path: Path, device: str
    ) -> Tuple[Dict[str, torch.Tensor], None]:
        from safetensors.torch import load_file

        sd = load_file(str(path), device=device)
        return sd, None

    def _load_torch(
        self, path: Path, device: str
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, Any]]]:
        try:
            obj = torch.load(str(path), map_location=device, weights_only=True)
        except Exception:
            obj = torch.load(str(path), map_location=device, weights_only=False)

        if not isinstance(obj, dict):
            raise RuntimeError(f"Unexpected payload type in {path}: {type(obj)}")

        embedded_cfg: Optional[Dict[str, Any]] = None
        if isinstance(obj.get("config"), dict):
            embedded_cfg = obj["config"]

        for key in ("model_state_dict", "state_dict", "model", "weights"):
            candidate = obj.get(key)
            if isinstance(candidate, dict) and candidate:
                sample = next(iter(candidate.values()), None)
                if isinstance(sample, torch.Tensor):
                    return candidate, embedded_cfg

        if obj and isinstance(next(iter(obj.values()), None), torch.Tensor):
            return obj, embedded_cfg

        raise RuntimeError(f"No tensor state dict found in {path}")


class ConfigReconstructor:
    """Reconstructs a TopoGPT2Config from weights, inferring every dimension."""

    def reconstruct(
        self,
        mod: Any,
        state_dict: Dict[str, torch.Tensor],
        embedded: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Return a TopoGPT2Config matching the loaded weights.

        Prefers the embedded config (saved by the scaler) but falls back
        to full inference from tensor shapes so that original checkpoints
        work without any embedded metadata.
        """
        if embedded is not None:
            known = set(mod.TopoGPT2Config.__dataclass_fields__.keys())
            filtered = {k: v for k, v in embedded.items() if k in known}
            try:
                return mod.TopoGPT2Config(**filtered)
            except Exception:
                pass
        return self._infer(mod, state_dict)

    def _infer(self, mod: Any, sd: Dict[str, torch.Tensor]) -> Any:
        te = sd["token_embed.weight"]
        vocab = int(te.shape[0])
        d_model = int(te.shape[1])

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


class TokenizerFactory:
    """Builds and caches BPETokenizer instances."""

    _instance: Optional[Any] = None

    def get(self, mod: Any) -> Any:
        """Return a shared tokenizer instance (built once per process)."""
        if self._instance is None:
            TokenizerFactory._instance = mod.BPETokenizer("gpt2")
        return self._instance


class GenerationEngine:
    """Autoregressive generation with top-k, top-p, and repetition penalty."""

    def __init__(self, cfg: SamplingConfig, device: str) -> None:
        self._cfg = cfg
        self._device = device

    def generate(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
    ) -> Tuple[str, int, float]:
        """Run generation and return (full_text, n_new_tokens, elapsed_s).

        Uses the model's built-in .generate() when repetition_penalty == 1.0
        and top_p == 1.0, otherwise falls back to a manual loop that supports
        the full sampling configuration.
        """
        torch.manual_seed(self._cfg.seed)
        ids = tokenizer.encode(prompt)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self._device)

        t0 = time.perf_counter()
        if self._cfg.repetition_penalty == 1.0 and self._cfg.top_p == 1.0:
            output = self._fast_generate(model, input_ids)
        else:
            output = self._manual_generate(model, input_ids)
        elapsed = time.perf_counter() - t0

        n_new = output.shape[1] - input_ids.shape[1]
        text = tokenizer.decode(output[0].tolist())
        return text, n_new, elapsed

    def _fast_generate(
        self, model: Any, input_ids: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            return model.generate(
                input_ids,
                max_new_tokens=self._cfg.max_new_tokens,
                temperature=self._cfg.temperature,
                top_k=self._cfg.top_k,
            )

    def _manual_generate(
        self, model: Any, input_ids: torch.Tensor
    ) -> torch.Tensor:
        generated = input_ids.clone()
        past_kvs = None

        for _ in range(self._cfg.max_new_tokens):
            ctx = generated if past_kvs is None else generated[:, -1:]
            with torch.no_grad():
                logits, _aux, past_kvs = model(ctx, past_kvs=past_kvs)

            logits = logits[:, -1, :].float()

            if self._cfg.repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(
                    logits, generated, self._cfg.repetition_penalty
                )

            logits = logits / max(self._cfg.temperature, 1e-8)

            if self._cfg.top_k > 0:
                top_vals, _ = torch.topk(
                    logits, min(self._cfg.top_k, logits.size(-1))
                )
                logits[logits < top_vals[:, -1:]] = float("-inf")

            if self._cfg.top_p < 1.0:
                logits = self._apply_top_p(logits, self._cfg.top_p)

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_tok], dim=1)

            eos_id = 50256
            if (next_tok == eos_id).all():
                break

        return generated

    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor,
        generated: torch.Tensor,
        penalty: float,
    ) -> torch.Tensor:
        unique_ids = generated[0].unique()
        score = logits[0, unique_ids]
        score = torch.where(score < 0, score * penalty, score / penalty)
        logits[0, unique_ids] = score
        return logits

    @staticmethod
    def _apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum_probs - F.softmax(sorted_logits, dim=-1) > p
        sorted_logits[remove] = float("-inf")
        return logits.scatter(1, sorted_idx, sorted_logits)


@dataclass
class ModelResult:
    """Result from running one checkpoint."""

    checkpoint: str
    label: str
    d_model: int
    n_layers: int
    n_heads: int
    n_params: int
    torus: str
    output_text: str
    new_tokens: int
    elapsed_s: float
    tokens_per_second: float
    error: Optional[str] = None


class MultiInferenceRunner:
    """Runs one prompt through every checkpoint and collects results."""

    def __init__(self, cfg: RunConfig) -> None:
        self._cfg = cfg
        self._log = _setup_logger("MultiInference", cfg.log_level)
        self._importer = ModuleImporter()
        self._discovery = CheckpointDiscovery()
        self._loader = CheckpointLoader()
        self._reconstructor = ConfigReconstructor()
        self._tokenizer_factory = TokenizerFactory()

    def run(self) -> List[ModelResult]:
        """Execute the full multi-model inference pipeline."""
        mod = self._importer.load(self._cfg.script)
        tokenizer = self._tokenizer_factory.get(mod)
        checkpoints = self._discovery.resolve(self._cfg.checkpoints)

        self._log.info("=" * 72)
        self._log.info("TopoGPT2 Multi-Model Inference")
        self._log.info("Prompt: %s", self._cfg.prompt[:80])
        self._log.info("Checkpoints found: %d", len(checkpoints))
        self._log.info("=" * 72)

        results: List[ModelResult] = []
        engine = GenerationEngine(self._cfg.sampling, self._cfg.device)

        for idx, ckpt_path in enumerate(checkpoints):
            self._log.info(
                "[%d/%d] %s", idx + 1, len(checkpoints), ckpt_path.name
            )
            result = self._run_one(ckpt_path, mod, tokenizer, engine)
            results.append(result)
            if result.error:
                self._log.error("  FAILED: %s", result.error)
            else:
                self._log.info(
                    "  D_MODEL=%d  params=%s  torus=%s  "
                    "tokens=%d  %.2f s  (%.1f tok/s)",
                    result.d_model,
                    _fmt_params(result.n_params),
                    result.torus,
                    result.new_tokens,
                    result.elapsed_s,
                    result.tokens_per_second,
                )

        return results

    def _run_one(
        self,
        ckpt_path: Path,
        mod: Any,
        tokenizer: Any,
        engine: GenerationEngine,
    ) -> ModelResult:
        label = self._make_label(ckpt_path)
        try:
            state_dict, embedded_cfg = self._loader.load(
                ckpt_path, self._cfg.device
            )
            model_cfg = self._reconstructor.reconstruct(
                mod, state_dict, embedded_cfg
            )
            model_cfg.DEVICE = self._cfg.device

            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = mod.TopoGPT2(model_cfg)

            model.to(self._cfg.device)
            missing, unexpected = model.load_state_dict(
                state_dict, strict=False
            )
            if unexpected:
                self._log.debug("Unexpected keys in %s: %s", label, unexpected[:4])

            model.eval()
            n_params = sum(p.numel() for p in model.parameters())

            text, n_new, elapsed = engine.generate(
                model, tokenizer, self._cfg.prompt
            )

            new_text = text[len(self._cfg.prompt):]
            tps = n_new / elapsed if elapsed > 0 else 0.0

            result = ModelResult(
                checkpoint=str(ckpt_path),
                label=label,
                d_model=int(model_cfg.D_MODEL),
                n_layers=int(model_cfg.N_LAYERS),
                n_heads=int(model_cfg.N_HEADS),
                n_params=n_params,
                torus=f"{model_cfg.TORUS_RADIAL_BINS}x{model_cfg.TORUS_ANGULAR_BINS}",
                output_text=new_text,
                new_tokens=n_new,
                elapsed_s=elapsed,
                tokens_per_second=tps,
            )

            del model, state_dict
            gc.collect()
            return result

        except Exception as exc:
            if "state_dict" in dir():
                del state_dict
            gc.collect()
            return ModelResult(
                checkpoint=str(ckpt_path),
                label=label,
                d_model=0, n_layers=0, n_heads=0, n_params=0,
                torus="?", output_text="", new_tokens=0,
                elapsed_s=0.0, tokens_per_second=0.0,
                error=str(exc),
            )

    @staticmethod
    def _make_label(path: Path) -> str:
        """Extract a short human-readable label from a checkpoint path."""
        name = path.stem
        for pat in (
            r"topogpt2_scaled_(D\d+)_\d+",
            r"checkpoint_phase\d+_training_epoch_(\d+)",
            r"checkpoint_epoch_(\d+)",
        ):
            m = re.search(pat, name)
            if m:
                return m.group(1)
        return name[:32]


class ResultRenderer:
    """Formats and prints inference results."""

    _COL_WIDTH = 70
    _SEPARATOR = "─"

    def render(self, results: List[ModelResult], prompt: str) -> None:
        """Print prompt header, per-model outputs, and comparison table."""
        self._print_prompt_header(prompt)
        for r in results:
            self._print_model_output(r)
        self._print_comparison_table(results)

    def _print_prompt_header(self, prompt: str) -> None:
        print()
        print("=" * 72)
        print("  PROMPT")
        print("=" * 72)
        print(prompt)
        print()

    def _print_model_output(self, r: ModelResult) -> None:
        sep = self._SEPARATOR * 72
        print(sep)
        if r.error:
            print(f"  [{r.label}]  ERROR: {r.error}")
            print(sep)
            return
        print(
            f"  [{r.label}]  "
            f"D_MODEL={r.d_model}  "
            f"params={_fmt_params(r.n_params)}  "
            f"torus={r.torus}  "
            f"layers={r.n_layers}"
        )
        print(
            f"  {r.new_tokens} tokens  "
            f"{r.elapsed_s:.2f}s  "
            f"{r.tokens_per_second:.1f} tok/s"
        )
        print(sep)
        print(r.output_text.strip())
        print()

    def _print_comparison_table(self, results: List[ModelResult]) -> None:
        ok = [r for r in results if not r.error]
        if not ok:
            return
        print("=" * 72)
        print("  COMPARISON TABLE")
        print("=" * 72)
        header = (
            f"{'label':<20} {'D_MODEL':>8} {'params':>12} "
            f"{'torus':>8} {'layers':>7} {'tok/s':>8} {'elapsed':>9}"
        )
        print(header)
        print(self._SEPARATOR * 72)
        for r in ok:
            row = (
                f"{r.label:<20} {r.d_model:>8} {_fmt_params(r.n_params):>12} "
                f"{r.torus:>8} {r.n_layers:>7} {r.tokens_per_second:>8.1f} "
                f"{r.elapsed_s:>8.2f}s"
            )
            print(row)
        print("=" * 72)

        if len(ok) > 1:
            base = ok[0]
            print()
            print(f"  Scaling relative to [{base.label}] (D_MODEL={base.d_model})")
            print(self._SEPARATOR * 72)
            for r in ok[1:]:
                param_ratio = r.n_params / max(base.n_params, 1)
                speed_ratio = r.tokens_per_second / max(base.tokens_per_second, 1e-9)
                print(
                    f"  [{r.label}]  "
                    f"params ×{param_ratio:.2f}  "
                    f"speed ×{speed_ratio:.2f}"
                )
            print()


def _fmt_params(n: int) -> str:
    """Format parameter counts as human-readable strings (25.1M, 147.5M)."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


class JsonExporter:
    """Saves results to a JSON file for later analysis."""

    def export(self, results: List[ModelResult], path: str, prompt: str) -> None:
        """Serialize results to JSON."""
        payload = {
            "prompt": prompt,
            "results": [
                {
                    "checkpoint": r.checkpoint,
                    "label": r.label,
                    "d_model": r.d_model,
                    "n_layers": r.n_layers,
                    "n_heads": r.n_heads,
                    "n_params": r.n_params,
                    "torus": r.torus,
                    "output_text": r.output_text,
                    "new_tokens": r.new_tokens,
                    "elapsed_s": r.elapsed_s,
                    "tokens_per_second": r.tokens_per_second,
                    "error": r.error,
                }
                for r in results
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {path}")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    p = argparse.ArgumentParser(
        description="TopoGPT2 Multi-Model Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
        help="Input prompt for all models",
    )
    p.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        metavar="PATH",
        help=(
            "One or more checkpoint files, directories, or glob patterns. "
            "Directories are searched recursively for .pt/.safetensors files."
        ),
    )
    p.add_argument(
        "--script",
        type=str,
        default="topogpt2_1.py",
        help="Path to topogpt2_1.py",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--max-new",
        type=int,
        default=200,
        dest="max_new_tokens",
        help="Maximum new tokens to generate per model",
    )
    p.add_argument("--temp", type=float, default=0.8, dest="temperature")
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus sampling threshold (1.0 = disabled)",
    )
    p.add_argument(
        "--rep-penalty",
        type=float,
        default=1.1,
        dest="repetition_penalty",
        help="Repetition penalty (1.0 = disabled)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--save-json",
        type=str,
        default="",
        metavar="FILE",
        help="Write results to a JSON file",
    )
    p.add_argument(
        "--log",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        dest="log_level",
    )
    return p


def config_from_args(args: argparse.Namespace) -> RunConfig:
    """Build a RunConfig from parsed CLI arguments."""
    return RunConfig(
        prompt=args.prompt,
        checkpoints=args.checkpoints,
        script=args.script,
        device=args.device,
        log_level=args.log_level,
        save_json=args.save_json,
        sampling=SamplingConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
        ),
    )


def main() -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    cfg = config_from_args(args)

    runner = MultiInferenceRunner(cfg)
    results = runner.run()

    renderer = ResultRenderer()
    renderer.render(results, cfg.prompt)

    if cfg.save_json:
        JsonExporter().export(results, cfg.save_json, cfg.prompt)

    failed = [r for r in results if r.error]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
