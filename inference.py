#!/usr/bin/env python3
"""
Inference module for TopoGPT2.
Handles model instantiation, checkpoint weight alignment, autoregressive generation,
and CLI execution following SOLID principles and scientific engineering standards.
"""
import torch
import torch.nn.functional as F
import importlib.util
import sys
import os
import argparse
import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from safetensors.torch import load_file

def _load_source_module(path: str):
    module_name = "topogpt2_source"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

@dataclass
class InferenceConfig:
    """Parametric configuration container for inference execution."""
    model_scale: str = "small"
    checkpoint_dir: str = "checkpoints_topogpt2"
    checkpoint_name: str = "best"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_k: int = 40
    repetition_penalty: float = 1.1
    batch_size: int = 1
    log_level: str = "INFO"
    source_file: str = "app.py"
    prompt: str = "Once upon a time"
    seed: int = 42
    dtype_override: str = "auto"

class CheckpointInspector:
    """Reads safetensors metadata to align architecture configuration."""
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def inspect_kq_head_count(self, checkpoint_path: str, d_model: int, n_heads: int) -> int:
        sd = load_file(checkpoint_path, device="cpu")
        key = "layers.0.attn.k_proj.weight"
        if key not in sd:
            self.logger.warning("Checkpoint missing k_proj weight. Falling back to configuration.")
            return 0
        k_dim = sd[key].shape[0]
        d_head = d_model // n_heads
        if d_head == 0:
            return 0
        return k_dim // d_head

    def patch_config(self, config: InferenceConfig, source_module: Any) -> Any:
        preset = self._resolve_preset(config.model_scale)
        d_model = preset["D_MODEL"]
        n_heads = preset["N_HEADS"]
        ckpt_path = os.path.join(config.checkpoint_dir, config.checkpoint_name, "model.safetensors")
        n_kv_heads = self.inspect_kq_head_count(ckpt_path, d_model, n_heads)
        cfg_cls = getattr(source_module, "TopoGPT2Config")
        aligned_cfg = cfg_cls(
            SCALE=config.model_scale,
            DEVICE=config.device,
            RANDOM_SEED=config.seed,
            CHECKPOINT_DIR=config.checkpoint_dir,
            N_KV_HEADS=n_kv_heads if n_kv_heads > 0 else 0,
        )
        tokenizer_cls = getattr(source_module, "BPETokenizer")
        tokenizer = tokenizer_cls("gpt2")
        aligned_cfg.VOCAB_SIZE = tokenizer.vocab_size
        return aligned_cfg

    def _resolve_preset(self, scale: str) -> Dict[str, int]:
        presets = {
            "micro": {"D_MODEL": 64, "N_HEADS": 4},
            "small": {"D_MODEL": 256, "N_HEADS": 8},
            "medium": {"D_MODEL": 512, "N_HEADS": 8},
            "gpt2": {"D_MODEL": 768, "N_HEADS": 12},
        }
        return presets.get(scale, presets["small"])

class ModelLoader:
    """Loads weights from safetensors and binds to the architectural graph."""
    def __init__(self, checkpoint_name: str, logger: logging.Logger):
        self.checkpoint_name = checkpoint_name
        self.logger = logger

    def load_model(self, config: Any, source_module: Any) -> Any:
        model_cls = getattr(source_module, "TopoGPT2")
        model = model_cls(config)
        model.to(config.DEVICE)
        model.eval()
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, self.checkpoint_name, "model.safetensors")
        if not os.path.exists(ckpt_path):
            self.logger.error("Checkpoint not found at %s", ckpt_path)
            sys.exit(1)
        self.logger.info("Loading weights from %s", ckpt_path)
        sd = load_file(ckpt_path, device=config.DEVICE)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            self.logger.info("Missing keys restored via weight tying: %s", missing)
        if unexpected:
            self.logger.info("Unexpected keys ignored: %s", unexpected)
        self.logger.info("Model successfully loaded and evaluated.")
        return model

class GenerationEngine:
    """Handles autoregressive token generation with controlled sampling."""
    def __init__(self, config: InferenceConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def generate(self, model: Any, tokenizer: Any, prompt_text: str) -> str:
        input_ids = torch.tensor([tokenizer.encode(prompt_text)], dtype=torch.long, device=self.config.device)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_k=self.config.top_k
            )
        return tokenizer.decode(output_ids[0].tolist())

    def sample_logits(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits / max(self.config.temperature, 1e-8)
        if self.config.top_k > 0:
            topk_values, _ = torch.topk(logits, min(self.config.top_k, logits.size(-1)))
            logits[logits < topk_values[:, -1:]] = float("-inf")
        probabilities = F.softmax(logits, dim=-1)
        return torch.multinomial(probabilities, num_samples=1)

class InferenceRunner:
    """Orchestrates execution flow for prompt processing."""
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("TopoGPT2Inference")
        logger.setLevel(getattr(logging, self.config.log_level.upper(), logging.INFO))
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            logger.addHandler(handler)
        return logger

    def run(self) -> None:
        source = _load_source_module(self.config.source_file)
        inspector = CheckpointInspector(self.logger)
        aligned_config = inspector.patch_config(self.config, source)
        loader = ModelLoader(self.config.checkpoint_name, self.logger)
        model = loader.load_model(aligned_config, source)
        tokenizer = getattr(source, "BPETokenizer")("gpt2")
        generator = GenerationEngine(self.config, self.logger)
        self.logger.info("Generating response for prompt: %s", self.config.prompt)
        start_time = time.time()
        output_text = generator.generate(model, tokenizer, self.config.prompt)
        elapsed = time.time() - start_time
        self.logger.info("Generation completed in %.2f seconds", elapsed)
        self._print_result(self.config.prompt, output_text)

    def _print_result(self, prompt: str, output: str) -> None:
        print("\n" + "=" * 70)
        print("PROMPT:")
        print(prompt)
        print("\nGENERATED OUTPUT:")
        print(output[len(prompt):])
        print("=" * 70)

def parse_arguments() -> InferenceConfig:
    parser = argparse.ArgumentParser(description="TopoGPT2 Inference Engine")
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--scale", type=str, default="small", choices=["micro", "small", "medium", "gpt2"])
    parser.add_argument("--max-new", type=int, default=128)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints_topogpt2")
    parser.add_argument("--ckpt-name", type=str, default="best")
    parser.add_argument("--source", type=str, default="topogpt2.1.py")
    parser.add_argument("--log", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    return InferenceConfig(
        model_scale=args.scale,
        checkpoint_dir=args.ckpt_dir,
        checkpoint_name=args.ckpt_name,
        device=args.device,
        max_new_tokens=args.max_new,
        temperature=args.temp,
        top_k=args.top_k,
        prompt=args.prompt,
        log_level=args.log,
        source_file=args.source,
    )

if __name__ == "__main__":
    inference_config = parse_arguments()
    runner = InferenceRunner(inference_config)
    runner.run()