#!/usr/bin/env python3
"""
Reinforcement Learning module for TopoGPT2.
Implements PPO-based alignment with mechanistic interpretability metrics as reward signals.
Supports chatbot/agent conversion via reward modeling and policy optimization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file as st_save, load_file as st_load
import numpy as np
import math
import os
import sys
import time
import json
import logging
import argparse
import importlib.util
from datetime import datetime
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass, field
from collections import deque
from enum import Enum, auto

@dataclass
class RLConfig:
    """Parametric configuration for RL alignment of TopoGPT2."""
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    RANDOM_SEED: int = 42
    USE_AMP: bool = True
    PPO_CLIP_EPS: float = 0.2
    PPO_VALUE_COEF: float = 0.5
    PPO_ENTROPY_COEF: float = 0.01
    PPO_GAMMA: float = 0.99
    PPO_GAE_LAMBDA: float = 0.95
    PPO_EPOCHS: int = 4
    PPO_MINIBATCH_SIZE: int = 32
    PPO_MAX_GRAD_NORM: float = 0.5
    REWARD_MODEL_SCALE: str = 'small'
    REWARD_HIDDEN_DIM: int = 128
    REWARD_DROPOUT: float = 0.1
    REWARD_LR: float = 1e-4
    REWARD_WEIGHT_DECAY: float = 0.01
    KL_COEF: float = 0.1
    KL_TARGET: float = 0.02
    KL_HORIZON: int = 2048
    RESPONSE_MAX_LEN: int = 256
    PROMPT_MAX_LEN: int = 128
    BATCH_SIZE: int = 4
    GRAD_ACCUM_STEPS: int = 4
    LOG_INTERVAL_STEPS: int = 50
    CHECKPOINT_DIR: str = 'checkpoints_rl'
    CHECKPOINT_INTERVAL_STEPS: int = 500
    REWARD_SIGNALS: List[str] = field(default_factory=lambda: ['coherence', 'relevance', 'novelty'])
    REWARD_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        'coherence': 0.4, 'relevance': 0.4, 'novelty': 0.2
    })
    MECHANISTIC_REWARD_WEIGHT: float = 0.1
    LC_REWARD_TARGET: float = 0.85
    SP_REWARD_TARGET: float = 0.3
    DELTA_REWARD_TARGET: float = 0.05
    TEMPERATURE: float = 0.7
    TOP_K: int = 40
    TOP_P: float = 0.9
    SAMPLES_PER_PROMPT: int = 4
    REF_MODEL_PATH: Optional[str] = None
    LOG_LEVEL: str = 'INFO'
    SOURCE_FILE: str = 'topogpt2.1.py'
    MODEL_PATH: str = ''

class RewardSignalType(Enum):
    """Enumeration of supported reward signal types."""
    COHERENCE = auto()
    RELEVANCE = auto()
    NOVELTY = auto()
    MECHANISTIC_LC = auto()
    MECHANISTIC_SP = auto()
    MECHANISTIC_DELTA = auto()
    KL_PENALTY = auto()
    LENGTH_PENALTY = auto()
    CUSTOM = auto()

class CheckpointPatcher:
    """Inspects checkpoint weights to align architecture configuration before instantiation."""
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def align_config(self, model_path: str, config: RLConfig, source_module: Any) -> Any:
        sd = st_load(model_path, device='cpu')
        key = 'layers.0.attn.k_proj.weight'
        cfg_cls = getattr(source_module, 'TopoGPT2Config')
        preset = self._resolve_preset(config.REWARD_MODEL_SCALE)
        d_model = preset["D_MODEL"]
        n_heads = preset["N_HEADS"]
        n_kv_heads = 0
        if key in sd:
            k_dim = sd[key].shape[0]
            d_head = d_model // n_heads
            if d_head > 0:
                n_kv_heads = k_dim // d_head
                self.logger.info("Checkpoint indicates N_KV_HEADS=%d (config default=%d)", n_kv_heads, cfg_cls(SCALE=config.REWARD_MODEL_SCALE, DEVICE=config.DEVICE).N_KV_HEADS if hasattr(cfg_cls(SCALE=config.REWARD_MODEL_SCALE, DEVICE=config.DEVICE), 'N_KV_HEADS') else 'computed')
        else:
            self.logger.warning("Could not find k_proj.weight in checkpoint. Using default GQA settings.")
        model_cfg = cfg_cls(SCALE=config.REWARD_MODEL_SCALE, DEVICE=config.DEVICE, N_KV_HEADS=n_kv_heads if n_kv_heads > 0 else 0)
        tokenizer_cls = getattr(source_module, 'BPETokenizer')
        tokenizer = tokenizer_cls('gpt2')
        model_cfg.VOCAB_SIZE = tokenizer.vocab_size
        return model_cfg

    def _resolve_preset(self, scale: str) -> Dict[str, int]:
        presets = {
            "micro": {"D_MODEL": 64, "N_HEADS": 4},
            "small": {"D_MODEL": 256, "N_HEADS": 8},
            "medium": {"D_MODEL": 512, "N_HEADS": 8},
            "gpt2": {"D_MODEL": 768, "N_HEADS": 12},
        }
        return presets.get(scale, presets["small"])

class MechanisticRewardCalculator:
    """Computes reward signals from mechanistic interpretability metrics."""
    def __init__(self, config: RLConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.lc_target = config.LC_REWARD_TARGET
        self.sp_target = config.SP_REWARD_TARGET
        self.delta_target = config.DELTA_REWARD_TARGET

    def compute_lc_reward(self, lc_value: float) -> float:
        deviation = abs(lc_value - self.lc_target)
        return max(0.0, 1.0 - deviation * 5.0)

    def compute_sp_reward(self, sp_value: float) -> float:
        deviation = abs(sp_value - self.sp_target)
        return max(0.0, 1.0 - deviation * 3.0)

    def compute_delta_reward(self, delta_value: float) -> float:
        if delta_value <= self.delta_target:
            return 1.0
        return max(0.0, 1.0 - (delta_value - self.delta_target) * 10.0)

    def compute_mechanistic_reward(self, metrics: Dict[str, float]) -> float:
        rewards = []
        if 'lc' in metrics:
            rewards.append(self.compute_lc_reward(metrics['lc']))
        if 'sp' in metrics:
            rewards.append(self.compute_sp_reward(metrics['sp']))
        if 'delta' in metrics:
            rewards.append(self.compute_delta_reward(metrics['delta']))
        return float(np.mean(rewards)) if rewards else 0.0

class RewardModel(nn.Module):
    """Lightweight reward model for scoring generated responses."""
    def __init__(self, config: RLConfig, vocab_size: int, d_model: int):
        super().__init__()
        self.config = config
        self.d_model = d_model
        self.hidden_dim = config.REWARD_HIDDEN_DIM
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.project_in = nn.Linear(d_model, self.hidden_dim)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.REWARD_DROPOUT),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            for _ in range(2)
        ])
        self.value_head = nn.Linear(self.hidden_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = x * attention_mask.unsqueeze(-1)
        x = x.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        x = self.project_in(x)
        for layer in self.layers:
            x = x + layer(x)
        return self.value_head(x).squeeze(-1)

class ExperienceBuffer:
    """Stores trajectories for PPO training with advantage computation."""
    def __init__(self, config: RLConfig, capacity: int = 4096):
        self.config = config
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        self.gamma = config.PPO_GAMMA
        self.gae_lambda = config.PPO_GAE_LAMBDA

    def add(self, experience: Dict[str, Any]) -> None:
        self.buffer.append(experience)

    def compute_advantages(self, values: torch.Tensor, rewards: torch.Tensor,
                          masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        gae = torch.zeros(1, device=values.device)
        for t in reversed(range(len(rewards))):
            next_value = values[t + 1] if t < len(rewards) - 1 else torch.zeros(1, device=values.device)
            next_nonterminal = masks[t + 1] if t < len(rewards) - 1 else torch.zeros(1, device=values.device)
            delta = rewards[t] + self.gamma * next_value * next_nonterminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_nonterminal * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
        return advantages, returns

    def sample_minibatches(self, batch_size: int) -> List[Dict[str, torch.Tensor]]:
        if len(self.buffer) < batch_size:
            return []
        indices = np.random.permutation(len(self.buffer))[:batch_size]
        minibatches = []
        for i in range(0, len(indices), self.config.PPO_MINIBATCH_SIZE):
            batch_idx = indices[i:i + self.config.PPO_MINIBATCH_SIZE]
            batch = [self.buffer[idx] for idx in batch_idx]
            minibatches.append(self._collate(batch))
        return minibatches

    def _collate(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        return {
            'input_ids': torch.stack([ex['input_ids'] for ex in batch]),
            'attention_mask': torch.stack([ex['attention_mask'] for ex in batch]),
            'actions': torch.stack([ex['actions'] for ex in batch]),
            'old_log_probs': torch.stack([ex['old_log_probs'] for ex in batch]),
            'values': torch.stack([ex['values'] for ex in batch]),
            'rewards': torch.stack([ex['rewards'] for ex in batch]),
            'masks': torch.stack([ex['masks'] for ex in batch]),
        }

    def clear(self) -> None:
        self.buffer.clear()

class ValueHead(nn.Module):
    """Scalar value estimation head attached to TopoGPT2 for PPO."""
    def __init__(self, d_model: int, hidden_dim: int = 64):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        last_hidden = hidden_states[:, -1, :]
        return self.project(last_hidden).squeeze(-1)

class PPOTrainer:
    """Proximal Policy Optimization trainer for TopoGPT2 alignment."""
    def __init__(self, policy_model: Any, config: RLConfig,
                 reward_model: Optional[RewardModel] = None,
                 ref_model: Optional[Any] = None,
                 logger: Optional[logging.Logger] = None):
        self.policy = policy_model
        self.config = config
        self.logger = logger or setup_logger('PPOTrainer', config.LOG_LEVEL)
        self.device = config.DEVICE
        self.value_head = ValueHead(config.D_MODEL).to(self.device)
        self.reward_model = reward_model
        self.ref_model = ref_model
        if self.ref_model is not None:
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False
        self.optimizer = torch.optim.AdamW(
            list(self.policy.parameters()) + list(self.value_head.parameters()),
            lr=config.REWARD_LR,
            weight_decay=config.REWARD_WEIGHT_DECAY,
            betas=(0.9, 0.95),
        )
        self.scaler = torch.amp.GradScaler(
            self.device.split(':')[0],
            enabled=config.USE_AMP and 'cuda' in self.device,
        )
        self.buffer = ExperienceBuffer(config)
        self.mechanistic_calculator = MechanisticRewardCalculator(config, self.logger)
        self.global_step = 0
        self.best_reward = -float('inf')

    def generate_with_policy(self, prompt_ids: torch.Tensor,
                            max_new_tokens: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self.policy.eval()
        with torch.no_grad():
            output_ids = self.policy.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=self.config.TEMPERATURE,
                top_k=self.config.TOP_K,
            )
        return output_ids

    def compute_kl_divergence(self, policy_logits: torch.Tensor,
                             ref_logits: torch.Tensor) -> torch.Tensor:
        policy_probs = F.softmax(policy_logits, dim=-1)
        ref_probs = F.softmax(ref_logits, dim=-1)
        kl = F.kl_div(
            F.log_softmax(policy_logits, dim=-1),
            ref_probs,
            reduction='none',
        ).sum(dim=-1)
        return kl.mean()

    def compute_reward(self, responses: torch.Tensor, prompts: torch.Tensor,
                      metrics: Optional[Dict[str, float]] = None) -> torch.Tensor:
        batch_size = responses.size(0)
        rewards = torch.zeros(batch_size, device=self.device)
        if self.reward_model is not None:
            full_ids = torch.cat([prompts, responses], dim=1)
            mask = torch.ones_like(full_ids, dtype=torch.float, device=self.device)
            rm_scores = self.reward_model(full_ids, mask)
            rewards += rm_scores * self.config.REWARD_WEIGHTS.get('relevance', 0.4)
        if metrics is not None and self.config.MECHANISTIC_REWARD_WEIGHT > 0:
            mech_reward = self.mechanistic_calculator.compute_mechanistic_reward(metrics)
            rewards += torch.tensor(
                mech_reward * self.config.MECHANISTIC_REWARD_WEIGHT,
                device=self.device,
            ).expand(batch_size)
        length_penalty = -0.01 * (responses.size(1) - prompts.size(1))
        rewards += torch.tensor(length_penalty, device=self.device).expand(batch_size)
        return rewards

    def collect_experience(self, prompts: torch.Tensor,
                          num_samples: int) -> List[Dict[str, Any]]:
        experiences = []
        for _ in range(num_samples):
            responses = self.generate_with_policy(prompts, self.config.RESPONSE_MAX_LEN)
            with torch.no_grad():
                logits, _, _ = self.policy(responses)
                log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
                actions = responses[:, 1:]
                token_log_probs = torch.gather(
                    log_probs, dim=-1, index=actions.unsqueeze(-1)
                ).squeeze(-1)
                values = self.value_head(
                    self.policy.token_embed(responses)
                ).unsqueeze(1).expand(-1, token_log_probs.size(1))
            metrics = self._extract_mechanistic_metrics(responses)
            rewards = self.compute_reward(responses, prompts, metrics)
            masks = torch.ones_like(token_log_probs, device=self.device)
            exp = {
                'input_ids': responses,
                'attention_mask': torch.ones_like(responses, device=self.device),
                'actions': actions,
                'old_log_probs': token_log_probs,
                'values': values,
                'rewards': rewards,
                'masks': masks,
                'metrics': metrics,
            }
            experiences.append(exp)
        return experiences

    def _extract_mechanistic_metrics(self, tokens: torch.Tensor) -> Dict[str, float]:
        metrics = {}
        try:
            with torch.no_grad():
                hidden = self.policy.token_embed(tokens)
                for name, param in self.policy.named_parameters():
                    if 'weight' in name and param.dim() >= 2:
                        w = param.detach().float()
                        if w.size(0) >= 4:
                            norms = w.norm(dim=1, keepdim=True).clamp(min=1e-8)
                            w_norm = w / norms
                            sim = (w_norm @ w_norm.t()).abs()
                            mask = ~torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
                            if mask.any():
                                lc = 1.0 - sim[mask].mean().clamp(0, 1).item()
                                metrics['lc'] = lc
                                corr = torch.corrcoef(w.reshape(w.size(0), -1).float()).nan_to_num(0.0).abs()
                                if corr.size(0) > 1:
                                    sp_mask = ~torch.eye(corr.size(0), dtype=torch.bool, device=corr.device)
                                    if sp_mask.any():
                                        metrics['sp'] = corr[sp_mask].mean().item()
                delta = 0.0
                for p in self.policy.parameters():
                    if p.numel() > 0 and p.is_floating_point():
                        margins = (p.data - p.data.round()).abs().max().item()
                        delta = max(delta, margins)
                metrics['delta'] = delta
        except Exception as e:
            self.logger.debug("Metric extraction error: %s", e)
        return metrics

    def ppo_update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.policy.train()
        self.value_head.train()
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        actions = batch['actions'].to(self.device)
        old_log_probs = batch['old_log_probs'].to(self.device)
        old_values = batch['values'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        masks = batch['masks'].to(self.device)
        advantages, returns = self.buffer.compute_advantages(
            old_values.squeeze(), rewards, masks
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        for _ in range(self.config.PPO_EPOCHS):
            with torch.amp.autocast(
                device_type=self.device.split(':')[0],
                enabled=self.config.USE_AMP and 'cuda' in self.device,
            ):
                logits, _, _ = self.policy(input_ids)
                logits = logits[:, :-1, :]
                log_probs = F.log_softmax(logits, dim=-1)
                new_log_probs = torch.gather(
                    log_probs, dim=-1, index=actions.unsqueeze(-1)
                ).squeeze(-1)
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.config.PPO_CLIP_EPS,
                                   1 + self.config.PPO_CLIP_EPS) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                values = self.value_head(self.policy.token_embed(input_ids))
                value_loss = F.mse_loss(values.squeeze(), returns)
                entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()
                kl_loss = torch.tensor(0.0, device=self.device)
                if self.ref_model is not None:
                    with torch.no_grad():
                        ref_logits, _, _ = self.ref_model(input_ids)
                    kl_loss = self.compute_kl_divergence(logits, ref_logits[:, :-1, :])
                loss = (
                    policy_loss
                    + self.config.PPO_VALUE_COEF * value_loss
                    - self.config.PPO_ENTROPY_COEF * entropy
                    + self.config.KL_COEF * kl_loss
                )
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(self.policy.parameters()) + list(self.value_head.parameters()),
                self.config.PPO_MAX_GRAD_NORM,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        self.global_step += 1
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'kl_divergence': kl_loss.item(),
            'mean_reward': rewards.mean().item(),
        }

    def train_step(self, prompts: torch.Tensor) -> Dict[str, float]:
        experiences = self.collect_experience(prompts, self.config.SAMPLES_PER_PROMPT)
        for exp in experiences:
            self.buffer.add(exp)
        metrics = {}
        minibatches = self.buffer.sample_minibatches(
            len(experiences) * self.config.BATCH_SIZE
        )
        for mb in minibatches:
            step_metrics = self.ppo_update(mb)
            for k, v in step_metrics.items():
                metrics[k] = metrics.get(k, 0.0) + v / len(minibatches)
        if self.global_step % self.config.LOG_INTERVAL_STEPS == 0:
            self.logger.info(
                "Step %d: reward=%.4f kl=%.4f entropy=%.4f",
                self.global_step, metrics.get('mean_reward', 0),
                metrics.get('kl_divergence', 0), metrics.get('entropy', 0)
            )
        if self.global_step % self.config.CHECKPOINT_INTERVAL_STEPS == 0:
            self._save_checkpoint()
        return metrics

    def _save_checkpoint(self) -> None:
        os.makedirs(self.config.CHECKPOINT_DIR, exist_ok=True)
        sd = {k: v.cpu() for k, v in self.policy.state_dict().items()
              if k != 'lm_head.weight'}
        sd.update({f'value.{k}': v.cpu() for k, v in self.value_head.state_dict().items()})
        st_save(sd, os.path.join(self.config.CHECKPOINT_DIR, 'ppo_policy.safetensors'))
        opt_sd = self.optimizer.state_dict()
        torch.save(opt_sd, os.path.join(self.config.CHECKPOINT_DIR, 'ppo_optimizer.pt'))
        state = {
            'global_step': self.global_step,
            'best_reward': self.best_reward,
            'timestamp': datetime.now().isoformat(),
        }
        with open(os.path.join(self.config.CHECKPOINT_DIR, 'ppo_state.json'), 'w') as f:
            json.dump(state, f, indent=2)

    def load_checkpoint(self, path: Optional[str] = None) -> bool:
        checkpoint_path = path or os.path.join(self.config.CHECKPOINT_DIR, 'ppo_policy.safetensors')
        if not os.path.exists(checkpoint_path):
            return False
        sd = st_load(checkpoint_path, device=self.device)
        policy_sd = {k: v for k, v in sd.items() if not k.startswith('value.')}
        value_sd = {k.replace('value.', ''): v for k, v in sd.items() if k.startswith('value.')}
        self.policy.load_state_dict(policy_sd, strict=False)
        self.value_head.load_state_dict(value_sd)
        opt_path = os.path.join(self.config.CHECKPOINT_DIR, 'ppo_optimizer.pt')
        if os.path.exists(opt_path):
            opt_sd = torch.load(opt_path, map_location=self.device, weights_only=False)
            self.optimizer.load_state_dict(opt_sd)
        state_path = os.path.join(self.config.CHECKPOINT_DIR, 'ppo_state.json')
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                state = json.load(f)
            self.global_step = state.get('global_step', 0)
            self.best_reward = state.get('best_reward', -float('inf'))
        return True

class ChatAgent:
    """High-level interface for RL-aligned TopoGPT2 as chatbot."""
    def __init__(self, policy_model: Any, config: RLConfig,
                 tokenizer: Any, logger: Optional[logging.Logger] = None):
        self.policy = policy_model
        self.config = config
        self.tokenizer = tokenizer
        self.logger = logger or setup_logger('ChatAgent', config.LOG_LEVEL)
        self.trainer: Optional[PPOTrainer] = None
        self.conversation_history: List[Dict[str, str]] = []

    def attach_trainer(self, trainer: PPOTrainer) -> None:
        self.trainer = trainer

    def respond(self, user_message: str, max_new_tokens: int = 128) -> str:
        self.conversation_history.append({'role': 'user', 'content': user_message})
        context = self._format_conversation()
        prompt_ids = torch.tensor(
            [self.tokenizer.encode(context)],
            dtype=torch.long,
            device=self.config.DEVICE,
        )
        with torch.no_grad():
            output_ids = self.policy.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=self.config.TEMPERATURE,
                top_k=self.config.TOP_K,
            )
        full_text = self.tokenizer.decode(output_ids[0].tolist())
        response = full_text[len(context):].strip()
        self.conversation_history.append({'role': 'assistant', 'content': response})
        return response

    def _format_conversation(self) -> str:
        lines = []
        for turn in self.conversation_history[-10:]:
            role = turn['role']
            content = turn['content']
            lines.append(f"{role.capitalize()}: {content}")
        lines.append("Assistant:")
        return ' '.join(lines)

    def train_on_feedback(self, user_message: str, response: str,
                         reward_score: float) -> Dict[str, float]:
        if self.trainer is None:
            return {}
        prompt_ids = torch.tensor(
            [self.tokenizer.encode(user_message)],
            dtype=torch.long,
            device=self.config.DEVICE,
        )
        response_ids = torch.tensor(
            [self.tokenizer.encode(response)],
            dtype=torch.long,
            device=self.config.DEVICE,
        )
        exp = {
            'input_ids': response_ids,
            'attention_mask': torch.ones_like(response_ids, device=self.config.DEVICE),
            'actions': response_ids[:, 1:],
            'old_log_probs': torch.zeros(response_ids.size(1) - 1, device=self.config.DEVICE),
            'values': torch.tensor([reward_score], device=self.config.DEVICE),
            'rewards': torch.tensor([reward_score], device=self.config.DEVICE),
            'masks': torch.ones(response_ids.size(1) - 1, device=self.config.DEVICE),
            'metrics': {},
        }
        self.trainer.buffer.add(exp)
        minibatches = self.trainer.buffer.sample_minibatches(self.config.BATCH_SIZE)
        metrics = {}
        for mb in minibatches:
            step_metrics = self.trainer.ppo_update(mb)
            metrics.update(step_metrics)
        return metrics

    def reset_conversation(self) -> None:
        self.conversation_history.clear()

def setup_logger(name: str, level: str = 'INFO') -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
        logger.addHandler(handler)
    return logger

def create_rl_agent_from_checkpoint(model_path: str, config: RLConfig,
                                    tokenizer: Any,
                                    logger: Optional[logging.Logger] = None) -> ChatAgent:
    spec = importlib.util.spec_from_file_location('topogpt2_source', config.SOURCE_FILE)
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    patcher = CheckpointPatcher(logger or setup_logger('CheckpointPatcher'))
    aligned_config = patcher.align_config(model_path, config, source)
    model_cls = getattr(source, 'TopoGPT2')
    policy = model_cls(aligned_config).to(config.DEVICE)
    sd = st_load(model_path, device=config.DEVICE)
    policy_sd = {k: v for k, v in sd.items() if not k.startswith('value.')}
    policy.load_state_dict(policy_sd, strict=False)
    return ChatAgent(policy, config, tokenizer, logger)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TopoGPT2 RL Alignment')
    parser.add_argument('--mode', choices=['train', 'chat', 'eval'], default='chat')
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--reward-model-path', type=str, default=None)
    parser.add_argument('--ref-model-path', type=str, default=None)
    parser.add_argument('--prompt', type=str, default='Hello, how are you?')
    parser.add_argument('--reward', type=float, default=1.0, help='Manual reward for feedback training')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--scale', type=str, default='small', choices=['micro', 'small', 'medium', 'gpt2'])
    args = parser.parse_args()
    logger = setup_logger('RLMain')
    spec = importlib.util.spec_from_file_location('topogpt2_source', 'topogpt2.1.py')
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    tokenizer_cls = getattr(source, 'BPETokenizer')
    tokenizer = tokenizer_cls('gpt2')
    config = RLConfig(
        DEVICE=args.device, 
        REWARD_MODEL_SCALE=args.scale, 
        SOURCE_FILE='topogpt2.1.py', 
        MODEL_PATH=args.model_path
    )
    agent = create_rl_agent_from_checkpoint(args.model_path, config, tokenizer, logger)
    if args.mode == 'chat':
        print(f"User: {args.prompt}")
        response = agent.respond(args.prompt)
        print(f"Assistant: {response}")
    elif args.mode == 'train':
        response = agent.respond(args.prompt)
        metrics = agent.train_on_feedback(args.prompt, response, args.reward)
        logger.info("Training metrics: %s", metrics)
    elif args.mode == 'eval':
        agent.policy.eval()
        with torch.no_grad():
            prompt_ids = torch.tensor(
                [tokenizer.encode(args.prompt)],
                dtype=torch.long,
                device=config.DEVICE,
            )
            output = agent.policy.generate(prompt_ids, max_new_tokens=100)
            print(tokenizer.decode(output[0].tolist()))