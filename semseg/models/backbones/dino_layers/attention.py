# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

from torch import Tensor
from torch import nn
import torch
import torch.nn.functional as F


logger = logging.getLogger("dinov2")


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention)")
# XFORMERS_AVAILABLE = False

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MulModalAttention(Attention):
    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv, modality_mask = self.qkv(x)
        m, b = modality_mask.shape
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        ############ self-attention ############
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        ############ cross-attention ############
        # q = q.reshape(m, b, self.num_heads, N, C // self.num_heads)
        # k = k.reshape(m, b, self.num_heads, N, C // self.num_heads)
        # v = v.reshape(m, b, self.num_heads, N, C // self.num_heads)
        
        # delta_x = torch.zeros([m, b, N, C], dtype=x.dtype, device=x.device)
        # for i in range(m):
        #     if modality_mask[i,0]:
        #         sel_q = q[i]
        #         for j in range(m):
        #             if i != j and modality_mask[j,0]:
        #                 sel_k = k[j]
        #                 sel_v = v[j]
        #                 attn = sel_q @ sel_k.transpose(-2, -1)
                        
        #                 attn = attn.softmax(dim=-1)
        #                 attn = self.attn_drop(attn)
                        
        #                 delta_x[i] += (attn @ sel_v).transpose(1, 2).reshape(b, N, C)                
        # delta_x = delta_x.reshape(m*b, N, C)
        # x = x + delta_x
        ############################################ 
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    
class MultiModalMemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv, modality_mask = self.qkv(x)
        m, b = modality_mask.shape
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        ############ cross-attention ############
        q = q.reshape(m, b, N, self.num_heads, C // self.num_heads)
        k = k.reshape(m, b, N, self.num_heads, C // self.num_heads)
        v = v.reshape(m, b, N, self.num_heads, C // self.num_heads)
        
        delta_x = torch.zeros([m, b, N, C], dtype=x.dtype, device=x.device)
        delta_count = torch.zeros([m, b, N, C], dtype=x.dtype, device=x.device)
        for i in range(m):
            if modality_mask[i,0]:
                sel_q = q[i]
                for j in range(m):
                    if i != j and modality_mask[j,0]:
                        sel_k = k[j]
                        sel_v = v[j]
                        delta_x[i] += memory_efficient_attention(sel_q, sel_k, sel_v, attn_bias=None).reshape([b, N, C])          
                        delta_count[i] += 1
        
        delta_x = delta_x / delta_count.clamp(min=1.0)
        delta_x = delta_x * modality_mask[:, :, None, None].float()
        delta_x = delta_x.reshape(m*b, N, C)
        alpha = 0.75
        x = alpha * x + (1-alpha) * delta_x
        ############################################

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

