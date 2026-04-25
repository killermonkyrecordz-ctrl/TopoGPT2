#!/usr/bin/env python3
"""
Quantized Inference Pipeline for TopoGPT2
Loads TopoGPT2, applies specified quantization format, and performs autoregressive text generation.
Architecture follows SOLID principles, is fully configuration-driven, and contains no magic numbers.
"""
import argparse
import json
import logging
import math
import os
import sys
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.data import DataLoader

try:
    import tiktoken
except ImportError:
    tiktoken = None


@dataclass
class InferenceConfig:
    """Centralized configuration for quantized inference. All parameters are explicitly defined."""
    MODEL_CONFIG: Dict[str, Any] = field(default_factory=lambda: {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'random_seed': 42,
        'scale': 'small',
        'vocab_size': 50257,
        'max_seq_len': 512,
        'd_model': 256,
        'n_heads': 8,
        'n_kv_heads': 0,
        'n_layers': 6,
        'dropout': 0.1,
        'moe_enabled': True,
        'n_experts': 4,
        'moe_top_k': 2,
        'moe_aux_loss_weight': 0.01,
        'torus_grid_size': 8,
        'torus_radial_bins': 2,
        'torus_angular_bins': 4,
        'spectral_latent_ratio': 0.5,
        'spectral_kernel_init_scale': 0.02,
        'num_spectral_layers': 2,
        'ae_recon_weight': 0.01,
        'gradient_checkpointing': True,
        't_init': 1.0,
        'rope_base': 10000,
    })
    QUANTIZATION_CONFIG: Dict[str, Any] = field(default_factory=lambda: {
        'format': 'float32',
        'bitnet_threshold': 0.33,
        'int4_symmetric': True,
        'int8_dynamic': True,
    })
    PATHS_CONFIG: Dict[str, Any] = field(default_factory=lambda: {
        'checkpoint_dir': 'checkpoints_topogpt2',
        'latest_checkpoint_subdir': 'latest',
        'model_filename': 'model.safetensors',
    })
    INFERENCE_CONFIG: Dict[str, Any] = field(default_factory=lambda: {
        'max_new_tokens': 200,
        'temperature': 0.8,
        'top_k': 50,
        'prompt': 'Once upon a time',
        'log_level': 'INFO',
        'tokenizer_encoding': 'gpt2',
        'eos_token_id': 50256,
    })
    SAFETY_CONFIG: Dict[str, Any] = field(default_factory=lambda: {
        'clamp_temperature_min': 1e-8,
    })

    def resolve_gqa(self) -> None:
        cfg = self.MODEL_CONFIG
        if cfg['n_kv_heads'] == 0:
            kv = max(1, cfg['n_heads'] // 4)
            while cfg['n_heads'] % kv != 0:
                kv -= 1
            cfg['n_kv_heads'] = kv
        elif cfg['n_kv_heads'] == -1:
            cfg['n_kv_heads'] = cfg['n_heads']
        assert cfg['n_heads'] % cfg['n_kv_heads'] == 0, "n_heads must be divisible by n_kv_heads"
        cfg['gqa_groups'] = cfg['n_heads'] // cfg['n_kv_heads']
        assert cfg['d_model'] % 4 == 0, "d_model must be divisible by 4 for quaternion operations"
        assert cfg['d_model'] % cfg['n_heads'] == 0, "d_model must be divisible by n_heads"
        cfg['d_quat'] = cfg['d_model'] // 4
        cfg['d_head'] = cfg['d_model'] // cfg['n_heads']
        cfg['spectral_latent_dim'] = max(16, int(cfg['d_model'] * cfg['spectral_latent_ratio']))
        cfg['n_torus_nodes'] = cfg['torus_radial_bins'] * cfg['torus_angular_bins']


class CheckpointInspector:
    """Inspects checkpoint state dict to dynamically resolve architecture parameters."""
    @staticmethod
    def inspect_and_patch(path: str, config: InferenceConfig) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        state_dict = load_file(path)
        cfg = config.MODEL_CONFIG
        
        # Resolve d_model from token embeddings or first linear layer
        if 'token_embed.weight' in state_dict:
            cfg['vocab_size'], cfg['d_model'] = state_dict['token_embed.weight'].shape
        elif 'layers.0.attn.q_proj.weight' in state_dict:
            cfg['d_model'] = state_dict['layers.0.attn.q_proj.weight'].shape[1]

        # Resolve n_heads and n_kv_heads from attention projections
        q_proj_key = 'layers.0.attn.q_proj.weight'
        k_proj_key = 'layers.0.attn.k_proj.weight'
        if q_proj_key in state_dict and k_proj_key in state_dict:
            q_out, d_model_ckpt = state_dict[q_proj_key].shape
            k_out, _ = state_dict[k_proj_key].shape
            assert d_model_ckpt == cfg['d_model'], "Inconsistent d_model across checkpoint layers"
            # Infer d_head: usually q_out / n_heads. We assume standard n_heads for d_model or infer from ratio.
            # Robust inference: d_head = q_out // n_heads. Try common n_heads values.
            possible_heads = [4, 8, 12, 16, 24, 32]
            d_head = None
            n_heads_inferred = None
            for h in possible_heads:
                if d_model_ckpt % h == 0 and q_out % h == 0:
                    candidate_d_head = q_out // h
                    if k_out % candidate_d_head == 0:
                        d_head = candidate_d_head
                        n_heads_inferred = h
                        break
            if d_head is None:
                # Fallback to gcd or default assumption
                d_head = math.gcd(q_out, k_out)
                n_heads_inferred = q_out // d_head
            
            cfg['n_heads'] = n_heads_inferred
            cfg['n_kv_heads'] = k_out // d_head

        # Resolve max_seq_len from RoPE cache if present
        rope_cos_key = 'layers.0.attn.rope.cos_cache'
        if rope_cos_key in state_dict:
            seq_len_ckpt = state_dict[rope_cos_key].shape[0]
            cfg['max_seq_len'] = max(cfg['max_seq_len'], seq_len_ckpt)
        
        # Resolve number of layers
        layer_keys = [k for k in state_dict.keys() if k.startswith('layers.') and '.weight' in k]
        if layer_keys:
            max_layer_idx = max(int(k.split('.')[1]) for k in layer_keys)
            cfg['n_layers'] = max_layer_idx + 1
            
        config.resolve_gqa()


class QuaternionOps:
    """Pure quaternion operations in PyTorch. Representation: [..., 4] -> [w, x, y, z]."""
    @staticmethod
    def hamilton_product(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        return torch.stack([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ], dim=-1)

    @staticmethod
    def normalize(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return q / (q.norm(dim=-1, keepdim=True) + eps)

    @staticmethod
    def conjugate(q: torch.Tensor) -> torch.Tensor:
        sign = q.new_tensor([1.0, -1.0, -1.0, -1.0])
        return q * sign

    @staticmethod
    def rotate_vector(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        zero = torch.zeros(*v.shape[:-1], 1, device=v.device, dtype=v.dtype)
        v_q = torch.cat([zero, v], dim=-1)
        q_c = QuaternionOps.conjugate(q)
        rotated = QuaternionOps.hamilton_product(
            QuaternionOps.hamilton_product(q, v_q), q_c
        )
        return rotated[..., 1:]


class QuaternionLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        assert in_features % 4 == 0 and out_features % 4 == 0
        self.in_q = in_features // 4
        self.out_q = out_features // 4
        self.Ww = nn.Linear(self.in_q, self.out_q, bias=False)
        self.Wx = nn.Linear(self.in_q, self.out_q, bias=False)
        self.Wy = nn.Linear(self.in_q, self.out_q, bias=False)
        self.Wz = nn.Linear(self.in_q, self.out_q, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        for w in [self.Ww, self.Wx, self.Wy, self.Wz]:
            nn.init.normal_(w.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.in_q
        xw, xx, xy, xz = x[..., :d], x[..., d:2*d], x[..., 2*d:3*d], x[..., 3*d:]
        ow = self.Ww(xw) - self.Wx(xx) - self.Wy(xy) - self.Wz(xz)
        ox = self.Ww(xx) + self.Wx(xw) + self.Wy(xz) - self.Wz(xy)
        oy = self.Ww(xy) - self.Wx(xz) + self.Wy(xw) + self.Wz(xx)
        oz = self.Ww(xz) + self.Wx(xy) - self.Wy(xx) + self.Wz(xw)
        out = torch.cat([ow, ox, oy, oz], dim=-1)
        return out + self.bias if self.bias is not None else out


class QuaternionSpectralLayer(nn.Module):
    def __init__(self, in_q: int, out_q: int, grid_h: int, grid_w: int, init_scale: float = 0.02):
        super().__init__()
        self.in_q = in_q
        self.out_q = out_q
        self.grid_h = grid_h
        self.grid_w = grid_w
        freq_h = grid_h
        freq_w = grid_w // 2 + 1
        for c in ('w', 'x', 'y', 'z'):
            self.register_parameter(f'kr_{c}', nn.Parameter(torch.randn(in_q, out_q, freq_h, freq_w) * init_scale))
            self.register_parameter(f'ki_{c}', nn.Parameter(torch.randn(in_q, out_q, freq_h, freq_w) * init_scale))

    def _kernel(self, c: str) -> torch.Tensor:
        return torch.complex(getattr(self, f'kr_{c}'), getattr(self, f'ki_{c}'))

    def _contract(self, W: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        return torch.einsum('iohw,bihw->bohw', W, X)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.in_q
        xw, xx, xy, xz = x[:, :q], x[:, q:2*q], x[:, 2*q:3*q], x[:, 3*q:]
        Xw = torch.fft.rfft2(xw, s=(self.grid_h, self.grid_w))
        Xx = torch.fft.rfft2(xx, s=(self.grid_h, self.grid_w))
        Xy = torch.fft.rfft2(xy, s=(self.grid_h, self.grid_w))
        Xz = torch.fft.rfft2(xz, s=(self.grid_h, self.grid_w))
        Ww, Wx, Wy, Wz = self._kernel('w'), self._kernel('x'), self._kernel('y'), self._kernel('z')
        C = {}
        for wc, W in (('w', Ww), ('x', Wx), ('y', Wy), ('z', Wz)):
            for xc, X in (('w', Xw), ('x', Xx), ('y', Xy), ('z', Xz)):
                C[(wc, xc)] = self._contract(W, X)
        Pw = C[('w','w')] - C[('x','x')] - C[('y','y')] - C[('z','z')]
        Px = C[('w','x')] + C[('x','w')] + C[('y','z')] - C[('z','y')]
        Py = C[('w','y')] - C[('x','z')] + C[('y','w')] + C[('z','x')]
        Pz = C[('w','z')] + C[('x','y')] - C[('y','x')] + C[('z','w')]
        ow = torch.fft.irfft2(Pw, s=(self.grid_h, self.grid_w))
        ox = torch.fft.irfft2(Px, s=(self.grid_h, self.grid_w))
        oy = torch.fft.irfft2(Py, s=(self.grid_h, self.grid_w))
        oz = torch.fft.irfft2(Pz, s=(self.grid_h, self.grid_w))
        return torch.cat([ow, ox, oy, oz], dim=1)


class SpectralAutoencoder(nn.Module):
    def __init__(self, config: InferenceConfig):
        super().__init__()
        cfg = config.MODEL_CONFIG
        d = cfg['d_model']
        d_lat = cfg['spectral_latent_dim']
        d_q = cfg['d_quat']
        r = cfg['torus_radial_bins']
        a = cfg['torus_angular_bins']
        init_s = cfg['spectral_kernel_init_scale']
        n_freq = d // 2 + 1
        self.enc_kr = nn.Parameter(torch.randn(n_freq) * init_s)
        self.enc_ki = nn.Parameter(torch.randn(n_freq) * init_s)
        self.dec_kr = nn.Parameter(torch.randn(n_freq) * init_s)
        self.dec_ki = nn.Parameter(torch.randn(n_freq) * init_s)
        self.enc_proj = QuaternionLinear(d, d_lat)
        self.dec_proj = QuaternionLinear(d_lat, d)
        self.torus_spectral = nn.ModuleList([
            QuaternionSpectralLayer(d_q, d_q, r, a, init_scale=init_s)
            for _ in range(cfg['num_spectral_layers'])
        ])
        self.act = nn.GELU()
        self.d_model = d

    def _filter1d(self, x: torch.Tensor, kr: torch.Tensor, ki: torch.Tensor) -> torch.Tensor:
        X = torch.fft.rfft(x, dim=-1)
        K = torch.complex(kr, ki)
        return torch.fft.irfft(X * K, n=self.d_model, dim=-1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_filt = self.act(self._filter1d(x, self.enc_kr, self.enc_ki))
        return self.enc_proj(x_filt)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.dec_proj(z)
        return self._filter1d(x, self.dec_kr, self.dec_ki)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        recon_loss = F.mse_loss(recon, x.detach())
        return z, recon_loss

    def process_torus_grid(self, grid: torch.Tensor) -> torch.Tensor:
        h = grid
        for layer in self.torus_spectral:
            h = self.act(layer(h))
        return h


class QuaternionTorusBrain(nn.Module):
    def __init__(self, d_model: int, config: InferenceConfig):
        super().__init__()
        cfg = config.MODEL_CONFIG
        self.d_model = d_model
        self.d_lat = cfg['spectral_latent_dim']
        self.d_q = d_model // 4
        self.n_radial = cfg['torus_radial_bins']
        self.n_angular = cfg['torus_angular_bins']
        self.n_nodes = cfg['n_torus_nodes']
        self.spectral_ae = SpectralAutoencoder(config)
        self.torus_proj = nn.Sequential(
            QuaternionLinear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 4),
        )
        self.node_embed = nn.Parameter(torch.randn(self.n_nodes, d_model) * 0.02)
        self.edge_quat = nn.Parameter(torch.randn(4, 4) * 0.1)
        self.node_net = QuaternionLinear(d_model, d_model)
        self.readout = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self._build_torus_graph()

    def _build_torus_graph(self):
        edges_i, edges_j, edge_type = [], [], []
        R, A = self.n_radial, self.n_angular
        for r in range(R):
            for a in range(A):
                n = r * A + a
                edges_i.append(n); edges_j.append(r * A + (a - 1) % A); edge_type.append(0)
                edges_i.append(n); edges_j.append(r * A + (a + 1) % A); edge_type.append(1)
                if r > 0:
                    edges_i.append(n); edges_j.append((r - 1) * A + a); edge_type.append(2)
                if r < R - 1:
                    edges_i.append(n); edges_j.append((r + 1) * A + a); edge_type.append(3)
        self.register_buffer('edges_i', torch.tensor(edges_i, dtype=torch.long))
        self.register_buffer('edges_j', torch.tensor(edges_j, dtype=torch.long))
        self.register_buffer('edge_type', torch.tensor(edge_type, dtype=torch.long))

    def _torus_soft_assign(self, phi1: torch.Tensor, phi2: torch.Tensor) -> torch.Tensor:
        BS = phi1.shape[0]
        device = phi1.device
        ang_pos = torch.linspace(-math.pi, math.pi, self.n_angular + 1, device=device)[:-1]
        rad_pos = torch.linspace(-math.pi, math.pi, self.n_radial + 1, device=device)[:-1]
        d_ang = torch.sin((phi1.unsqueeze(1) - ang_pos.unsqueeze(0)) / 2).pow(2)
        d_rad = torch.sin((phi2.unsqueeze(1) - rad_pos.unsqueeze(0)) / 2).pow(2)
        d_torus = d_rad.unsqueeze(2) + d_ang.unsqueeze(1)
        d_flat = d_torus.view(BS, -1)
        return torch.softmax(-d_flat / 0.3, dim=-1)

    def _message_passing(self, node_feat: torch.Tensor) -> torch.Tensor:
        BS = node_feat.shape[0]
        n_edges = self.edges_i.shape[0]
        d_q = self.d_q
        eq = QuaternionOps.normalize(self.edge_quat)
        src_feat = node_feat[:, self.edges_j, :]
        edge_q = eq[self.edge_type].unsqueeze(0).unsqueeze(2).expand(BS, -1, d_q, -1)
        src_q = src_feat.view(BS, n_edges, d_q, 4)
        msg_rot = QuaternionOps.hamilton_product(edge_q, src_q)
        msg_rot = msg_rot.view(BS, n_edges, self.d_model)
        agg = torch.zeros_like(node_feat)
        dst_idx = self.edges_i.view(1, n_edges, 1).expand(BS, -1, self.d_model)
        agg.scatter_add_(1, dst_idx, msg_rot)
        return self.node_net(node_feat + agg)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, D = x.shape
        device = x.device
        x_flat = x.reshape(B * S, D)
        z, recon_loss = self.spectral_ae(x_flat)
        coords = self.torus_proj(x_flat)
        phi1 = math.pi * torch.tanh(coords[:, 0])
        phi2 = math.pi * torch.tanh(coords[:, 1])
        attn_w = self._torus_soft_assign(phi1, phi2)
        nodes = (
            attn_w.unsqueeze(-1) * self.node_embed.unsqueeze(0) +
            attn_w.unsqueeze(-1) * x_flat.unsqueeze(1)
        )
        grid = nodes.view(B * S, self.n_radial, self.n_angular, D)
        grid = grid.permute(0, 3, 1, 2)
        d_q = self.d_q
        grid_q = grid.view(B * S, 4, d_q, self.n_radial, self.n_angular)
        grid_q = grid_q.permute(0, 1, 2, 3, 4).reshape(B * S, 4 * d_q, self.n_radial, self.n_angular)
        grid_spec = self.spectral_ae.process_torus_grid(grid_q)
        grid_back = grid_spec.view(B * S, 4, d_q, self.n_radial, self.n_angular)
        grid_back = grid_back.permute(0, 3, 4, 1, 2).reshape(B * S, self.n_nodes, D)
        nodes_mp = self._message_passing(grid_back)
        out_flat = (attn_w.unsqueeze(-1) * nodes_mp).sum(dim=1)
        out_flat = self.readout(out_flat)
        output = out_flat.reshape(B, S, D)
        return output, recon_loss


class RotaryEmbedding(nn.Module):
    def __init__(self, d_head: int, max_seq_len: int, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer('inv_freq', inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cache', emb.cos())
        self.register_buffer('sin_cache', emb.sin())

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        needed = offset + max(q.shape[2], k.shape[2])
        if needed > self.cos_cache.shape[0]:
            self._build_cache(needed * 2)
        sq, sk = q.shape[2], k.shape[2]
        cos_q = self.cos_cache[offset:offset + sq].unsqueeze(0).unsqueeze(0)
        sin_q = self.sin_cache[offset:offset + sq].unsqueeze(0).unsqueeze(0)
        cos_k = self.cos_cache[offset:offset + sk].unsqueeze(0).unsqueeze(0)
        sin_k = self.sin_cache[offset:offset + sk].unsqueeze(0).unsqueeze(0)
        q_rot = q * cos_q + self._rotate_half(q) * sin_q
        k_rot = k * cos_k + self._rotate_half(k) * sin_k
        return q_rot, k_rot


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, expansion: float = 8/3, dropout: float = 0.0):
        super().__init__()
        inner = max(4, int(d_model * expansion))
        inner = (inner + 3) // 4 * 4
        self.gate_proj = nn.Linear(d_model, inner, bias=False)
        self.up_proj = nn.Linear(d_model, inner, bias=False)
        self.down_proj = nn.Linear(inner, d_model, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.normal_(self.gate_proj.weight, std=0.02)
        nn.init.normal_(self.up_proj.weight, std=0.02)
        nn.init.normal_(self.down_proj.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class TopoMoEBrain(nn.Module):
    def __init__(self, d_model: int, config: InferenceConfig):
        super().__init__()
        cfg = config.MODEL_CONFIG
        self.d_model = d_model
        self.moe_enabled = cfg['moe_enabled']
        self.n_experts = cfg['n_experts']
        self.top_k = cfg['moe_top_k']
        self.aux_weight = cfg['moe_aux_loss_weight']
        self.shared_expert = QuaternionTorusBrain(d_model, config)
        if self.moe_enabled:
            self.experts = nn.ModuleList([
                SwiGLU(d_model, expansion=4/3, dropout=cfg['dropout'])
                for _ in range(self.n_experts)
            ])
            self.router = nn.Linear(d_model, self.n_experts, bias=False)
            nn.init.normal_(self.router.weight, std=0.02)

    def _route(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        N, D = x.shape
        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)
        top_k_probs, top_k_idx = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        flat_idx = top_k_idx.reshape(-1)
        flat_weights = top_k_probs.reshape(-1)
        token_indices = torch.arange(N, device=x.device, dtype=torch.long).unsqueeze(1).expand(-1, self.top_k).reshape(-1)
        expert_out = torch.zeros_like(x)
        for e in range(self.n_experts):
            expert_mask = (flat_idx == e)
            src_token_idx = token_indices[expert_mask]
            w = flat_weights[expert_mask].unsqueeze(-1).to(x.dtype)
            out_e = self.experts[e](x[src_token_idx])
            contrib = w * out_e
            expert_out.scatter_add_(0, src_token_idx.unsqueeze(1).expand_as(contrib), contrib)
        token_frac = router_probs.mean(dim=0)
        one_hot = F.one_hot(top_k_idx, self.n_experts).float()
        dispatch_frac = one_hot.mean(dim=(0, 1))
        aux_loss = self.n_experts * (token_frac * dispatch_frac).sum()
        return expert_out, aux_loss

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, D = x.shape
        shared_out, recon_loss = self.shared_expert(x)
        if not self.moe_enabled:
            return shared_out, recon_loss
        x_flat = x.reshape(B * S, D)
        expert_out, aux_loss = self._route(x_flat)
        expert_out = expert_out.reshape(B, S, D)
        output = shared_out + expert_out
        total_aux = recon_loss + self.aux_weight * aux_loss
        return output, total_aux


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, config: InferenceConfig):
        super().__init__()
        cfg = config.MODEL_CONFIG
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv = cfg['n_kv_heads']
        self.n_groups = cfg['gqa_groups']
        self.d_head = cfg['d_head']
        self.q_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv * self.d_head, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.d_head, max_seq_len=cfg['max_seq_len'], base=cfg['rope_base'])
        self.temperature = nn.Parameter(torch.tensor(cfg['t_init']))
        self.dropout_p = cfg['dropout'] if cfg['dropout'] > 0 else 0.0

    def forward(self, x: torch.Tensor, is_causal: bool = True, past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, S, D = x.shape
        Q = self.q_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(B, S, self.n_kv, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(B, S, self.n_kv, self.d_head).transpose(1, 2)
        past_len = past_kv[0].shape[2] if past_kv is not None else 0
        Q, K = self.rope(Q, K, seq_len=S, offset=past_len)
        if past_kv is not None:
            K = torch.cat([past_kv[0], K], dim=2)
            V = torch.cat([past_kv[1], V], dim=2)
        kv_cache = (K, V)
        K_full = K
        if self.n_groups > 1:
            K_full = K_full.repeat_interleave(self.n_groups, dim=1)
            V_exp = V.repeat_interleave(self.n_groups, dim=1)
        else:
            V_exp = V
        scale = (self.d_head ** -0.5) / self.temperature.abs().clamp(min=1e-6)
        out = F.scaled_dot_product_attention(
            Q, K_full, V_exp,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=(is_causal and past_kv is None),
            scale=scale.item(),
        )
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.o_proj(out), kv_cache


class TopoGPT2Layer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, config: InferenceConfig):
        super().__init__()
        cfg = config.MODEL_CONFIG
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, config)
        self.topo_brain = TopoMoEBrain(d_model, config)
        self.dropout = nn.Dropout(cfg['dropout'])
        self.use_ckpt = cfg['gradient_checkpointing']

    def _forward_impl(self, x: torch.Tensor, past_kv: Optional[Tuple] = None) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        attn_out, kv_cache = self.attn(self.norm1(x), past_kv=past_kv)
        x = x + self.dropout(attn_out)
        brain_out, aux_loss = self.topo_brain(self.norm2(x))
        x = x + self.dropout(brain_out)
        return x, aux_loss, kv_cache

    def forward(self, x: torch.Tensor, past_kv: Optional[Tuple] = None) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        return self._forward_impl(x, past_kv=past_kv)


class TopoGPT2(nn.Module):
    def __init__(self, config: InferenceConfig):
        super().__init__()
        cfg = config.MODEL_CONFIG
        self.config = config
        self.token_embed = nn.Embedding(cfg['vocab_size'], cfg['d_model'])
        nn.init.normal_(self.token_embed.weight, std=0.02)
        self.layers = nn.ModuleList([
            TopoGPT2Layer(cfg['d_model'], cfg['n_heads'], config)
            for _ in range(cfg['n_layers'])
        ])
        self.final_norm = RMSNorm(cfg['d_model'])
        self.lm_head = nn.Linear(cfg['d_model'], cfg['vocab_size'], bias=False)
        self.lm_head.weight = self.token_embed.weight
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.lm_head:
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, token_ids: torch.Tensor, past_kvs: Optional[List[Optional[Tuple]]] = None) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple]]:
        x = self.token_embed(token_ids)
        total_aux = torch.tensor(0.0, device=x.device)
        new_kvs: List[Tuple] = []
        for i, layer in enumerate(self.layers):
            pkv = past_kvs[i] if past_kvs is not None else None
            x, al, kvc = layer(x, past_kv=pkv)
            total_aux = total_aux + al
            new_kvs.append(kvc)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, total_aux / len(self.layers), new_kvs

    @torch.no_grad()
    def generate(self, token_ids: torch.Tensor, max_new_tokens: int, temperature: float, top_k: int, eos_token_id: int) -> torch.Tensor:
        self.eval()
        cfg = self.config.MODEL_CONFIG if hasattr(self, 'config') else self.config
        ctx = token_ids[:, -cfg['max_seq_len']:]
        logits, _, past_kvs = self(ctx)
        logits = logits[:, -1, :] / max(temperature, 1e-8)
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, -1:]] = float('-inf')
        next_tok = torch.multinomial(F.softmax(logits, dim=-1), 1)
        token_ids = torch.cat([token_ids, next_tok], dim=1)
        for _ in range(max_new_tokens - 1):
            logits, _, past_kvs = self(next_tok, past_kvs=past_kvs)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float('-inf')
            next_tok = torch.multinomial(F.softmax(logits, dim=-1), 1)
            token_ids = torch.cat([token_ids, next_tok], dim=1)
            if (next_tok == eos_token_id).all():
                break
        return token_ids


class BPETokenizer:
    def __init__(self, encoding: str = 'gpt2'):
        if tiktoken is None:
            raise ImportError("tiktoken is required for tokenization.")
        self.enc = tiktoken.get_encoding(encoding)
        self.vocab_size = self.enc.n_vocab

    def encode(self, text: str) -> List[int]:
        return self.enc.encode(text, allowed_special={'<|endoftext|>'})

    def decode(self, tokens: List[int]) -> str:
        return self.enc.decode(tokens)

    def eot_token(self) -> int:
        return self.enc.eot_token


class QuantizationFormat(Enum):
    BITNET = "bitnet"
    INT4 = "int4"
    INT8 = "int8"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"


class IQuantizer(ABC):
    @abstractmethod
    def quantize(self, model: nn.Module) -> nn.Module:
        pass

    @abstractmethod
    def get_format_name(self) -> str:
        pass

    @abstractmethod
    def get_bits_per_weight(self) -> float:
        pass


class BitNetQuantizer(IQuantizer):
    def __init__(self, config: InferenceConfig):
        self.threshold = config.QUANTIZATION_CONFIG['bitnet_threshold']

    def quantize(self, model: nn.Module) -> nn.Module:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.numel() == 0:
                    continue
                abs_mean = param.data.abs().mean()
                ternary_weights = torch.where(
                    param.data > self.threshold * abs_mean,
                    torch.ones_like(param.data),
                    torch.where(
                        param.data < -self.threshold * abs_mean,
                        -torch.ones_like(param.data),
                        torch.zeros_like(param.data)
                    )
                )
                param.data.copy_(ternary_weights)
        return model

    def get_format_name(self) -> str:
        return "BitNet (1.58-bit ternary)"

    def get_bits_per_weight(self) -> float:
        return 1.58


class INT4Quantizer(IQuantizer):
    def __init__(self, config: InferenceConfig):
        self.symmetric = config.QUANTIZATION_CONFIG['int4_symmetric']

    def quantize(self, model: nn.Module) -> nn.Module:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.numel() == 0:
                    continue
                w = param.data
                w_max = w.abs().max() + 1e-8
                scale = 7.0 / w_max
                w_scaled = w * scale
                w_quantized = torch.clamp(torch.round(w_scaled), -8, 7)
                w_dequantized = w_quantized / scale
                param.data.copy_(w_dequantized)
        return model

    def get_format_name(self) -> str:
        return "INT4 (4-bit integer)"

    def get_bits_per_weight(self) -> float:
        return 4.0


class INT8Quantizer(IQuantizer):
    def __init__(self, config: InferenceConfig):
        pass

    def quantize(self, model: nn.Module) -> nn.Module:
        return torch.quantization.quantize_dynamic(model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8)

    def get_format_name(self) -> str:
        return "INT8 (8-bit integer)"

    def get_bits_per_weight(self) -> float:
        return 8.0


class Float16Quantizer(IQuantizer):
    def __init__(self, config: InferenceConfig):
        pass

    def quantize(self, model: nn.Module) -> nn.Module:
        return model.half()

    def get_format_name(self) -> str:
        return "FLOAT16 (half precision)"

    def get_bits_per_weight(self) -> float:
        return 16.0


class BFloat16Quantizer(IQuantizer):
    def __init__(self, config: InferenceConfig):
        pass

    def quantize(self, model: nn.Module) -> nn.Module:
        return model.to(torch.bfloat16)

    def get_format_name(self) -> str:
        return "BFLOAT16 (brain float)"

    def get_bits_per_weight(self) -> float:
        return 16.0


class Float32Quantizer(IQuantizer):
    def __init__(self, config: InferenceConfig):
        pass

    def quantize(self, model: nn.Module) -> nn.Module:
        return model.float()

    def get_format_name(self) -> str:
        return "FLOAT32 (single precision)"

    def get_bits_per_weight(self) -> float:
        return 32.0


class Float64Quantizer(IQuantizer):
    def __init__(self, config: InferenceConfig):
        pass

    def quantize(self, model: nn.Module) -> nn.Module:
        return model.double()

    def get_format_name(self) -> str:
        return "FLOAT64 (double precision)"

    def get_bits_per_weight(self) -> float:
        return 64.0


class QuantizerFactory:
    @staticmethod
    def create_quantizer(fmt: QuantizationFormat, config: InferenceConfig) -> IQuantizer:
        quantizers = {
            QuantizationFormat.BITNET: BitNetQuantizer,
            QuantizationFormat.INT4: INT4Quantizer,
            QuantizationFormat.INT8: INT8Quantizer,
            QuantizationFormat.FLOAT16: Float16Quantizer,
            QuantizationFormat.BFLOAT16: BFloat16Quantizer,
            QuantizationFormat.FLOAT32: Float32Quantizer,
            QuantizationFormat.FLOAT64: Float64Quantizer,
        }
        quantizer_class = quantizers.get(fmt)
        if quantizer_class is None:
            raise ValueError(f"Unsupported quantization format: {fmt}")
        return quantizer_class(config)


class ModelLoader:
    def __init__(self, config: InferenceConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def load_checkpoint(self) -> TopoGPT2:
        checkpoint_path = os.path.join(
            self.config.PATHS_CONFIG['checkpoint_dir'],
            self.config.PATHS_CONFIG['latest_checkpoint_subdir'],
            self.config.PATHS_CONFIG['model_filename']
        )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        # Dynamically adjust config to match checkpoint architecture
        self.logger.info("Inspecting checkpoint to resolve architecture parameters...")
        CheckpointInspector.inspect_and_patch(checkpoint_path, self.config)
        self.logger.info(f"Resolved architecture: d_model={self.config.MODEL_CONFIG['d_model']}, "
                         f"n_kv_heads={self.config.MODEL_CONFIG['n_kv_heads']}, "
                         f"max_seq_len={self.config.MODEL_CONFIG['max_seq_len']}")

        model = TopoGPT2(self.config)
        state_dict = load_file(checkpoint_path)
        
        # Safe loading: filter out shape mismatches for dynamic buffers
        model_state = model.state_dict()
        compatible_dict = {}
        mismatched = []
        for k, v in state_dict.items():
            if k in model_state and v.shape == model_state[k].shape:
                compatible_dict[k] = v
            elif k in model_state:
                mismatched.append(k)
                
        if mismatched:
            self.logger.warning(f"Skipping {len(mismatched)} mismatched keys (e.g., RoPE cache size): {mismatched[:5]}...")
            
        missing, unexpected = model.load_state_dict(compatible_dict, strict=False)
        if missing:
            self.logger.warning(f"Missing keys in state dict: {missing}")
        if unexpected:
            unexpected_real = [k for k in unexpected if k != 'lm_head.weight']
            if unexpected_real:
                self.logger.warning(f"Unexpected keys in state dict: {unexpected_real}")
        model.eval()
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return model


class InferenceEngine:
    def __init__(self, config: InferenceConfig, model: TopoGPT2, tokenizer: BPETokenizer):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = config.MODEL_CONFIG['device']
        self.model.to(self.device)

    def run_inference(self, prompt: str) -> str:
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        cfg = self.config.INFERENCE_CONFIG
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_tensor,
                max_new_tokens=cfg['max_new_tokens'],
                temperature=cfg['temperature'],
                top_k=cfg['top_k'],
                eos_token_id=cfg['eos_token_id']
            )
        output_text = self.tokenizer.decode(generated_ids[0].tolist())
        return output_text


class QuantizationInferencePipeline:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.logger = logging.getLogger("QuantizationInference")
        level = getattr(logging, config.INFERENCE_CONFIG['log_level'].upper(), logging.INFO)
        self.logger.setLevel(level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
        self.loader = ModelLoader(config, self.logger)

    def execute(self) -> str:
        self.logger.info("Initializing quantization inference pipeline.")
        fmt_str = self.config.QUANTIZATION_CONFIG['format'].lower()
        try:
            fmt = QuantizationFormat(fmt_str)
        except ValueError:
            self.logger.error(f"Invalid quantization format: {fmt_str}")
            raise
        model = self.loader.load_checkpoint()
        quantizer = QuantizerFactory.create_quantizer(fmt, self.config)
        self.logger.info(f"Applying {quantizer.get_format_name()} quantization ({quantizer.get_bits_per_weight()} bits/weight).")
        quantized_model = quantizer.quantize(model)
        tokenizer = BPETokenizer(self.config.INFERENCE_CONFIG['tokenizer_encoding'])
        engine = InferenceEngine(self.config, quantized_model, tokenizer)
        self.logger.info("Running inference.")
        result = engine.run_inference(self.config.INFERENCE_CONFIG['prompt'])
        self.logger.info("Inference complete.")
        return result


def parse_arguments() -> InferenceConfig:
    parser = argparse.ArgumentParser(description="Quantized Inference for TopoGPT2")
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_topogpt2')
    parser.add_argument('--quantization_format', type=str, default='float32',
                        choices=['bitnet', 'int4', 'int8', 'float16', 'bfloat16', 'float32', 'float64'])
    parser.add_argument('--prompt', type=str, default='Once upon a time')
    parser.add_argument('--max_new_tokens', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    args = parser.parse_args()
    config = InferenceConfig()
    config.resolve_gqa()
    config.PATHS_CONFIG['checkpoint_dir'] = args.checkpoint_dir
    config.QUANTIZATION_CONFIG['format'] = args.quantization_format
    config.INFERENCE_CONFIG['prompt'] = args.prompt
    config.INFERENCE_CONFIG['max_new_tokens'] = args.max_new_tokens
    config.INFERENCE_CONFIG['temperature'] = args.temperature
    config.INFERENCE_CONFIG['top_k'] = args.top_k
    config.INFERENCE_CONFIG['log_level'] = args.log_level
    if args.device:
        config.MODEL_CONFIG['device'] = args.device
    return config


def main() -> int:
    config = parse_arguments()
    try:
        pipeline = QuantizationInferencePipeline(config)
        output = pipeline.execute()
        print(output)
        return 0
    except Exception as e:
        logging.getLogger("QuantizationInference").error(f"Pipeline failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())