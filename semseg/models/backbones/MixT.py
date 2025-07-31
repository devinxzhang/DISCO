# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from torch.nn.init import xavier_uniform_, constant_

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

#from mmseg.utils import get_root_logger
#from mmcv.runner import load_checkpoint
import math
from typing import List, Tuple, Optional
from einops import rearrange, repeat
# from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
# from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated, LayerNorm
# from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
# from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined
# try:
#     from causal_conv1d import causal_conv1d_fn
# except ImportError:
#     causal_conv1d_fn = None
# import torch.distributed as dist
from mmcv.runner.base_module import BaseModule, ModuleList
from torch.utils.cpp_extension import load
# wkv_cuda = load(name="bi_wkv", sources=["semseg/models/backbones/cuda_new/bi_wkv.cpp", "semseg/models/backbones/cuda_new/bi_wkv_kernel.cu"],
#                 verbose=True, extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3', '-gencode arch=compute_86,code=sm_86'])
from .lora import LoRA, MultiModalLoRA_q, MultiModalLoRA_kv


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class MixVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])
        
        # self.r = 16
        # self.w_a = []
        # self.w_b = []
        # for i, block in enumerate(self.block1):
        #     w_q_linear = block.attn.q
        #     dim = w_q_linear.in_features
        #     i_w_a_linear_q, i_w_b_linear_q = self._create_lora_layer(dim, self.r)
        #     i_w_a_linear_v, i_w_b_linear_v = self._create_lora_layer(dim, self.r)
        #     d_w_a_linear_q, d_w_b_linear_q = self._create_lora_layer(dim, self.r)
        #     d_w_a_linear_v, d_w_b_linear_v = self._create_lora_layer(dim, self.r)
        #     e_w_a_linear_q, e_w_b_linear_q = self._create_lora_layer(dim, self.r)
        #     e_w_a_linear_v, e_w_b_linear_v = self._create_lora_layer(dim, self.r)
        #     l_w_a_linear_q, l_w_b_linear_q = self._create_lora_layer(dim, self.r)
        #     l_w_a_linear_v, l_w_b_linear_v = self._create_lora_layer(dim, self.r)
        #     self.w_a.extend([i_w_a_linear_q, i_w_a_linear_v, d_w_a_linear_q, d_w_a_linear_v, e_w_a_linear_q, e_w_a_linear_v, l_w_a_linear_q, l_w_a_linear_v])
        #     self.w_b.extend([i_w_b_linear_q, i_w_b_linear_v, d_w_b_linear_q, d_w_b_linear_v, e_w_b_linear_q, e_w_b_linear_v, l_w_b_linear_q, l_w_b_linear_v])
        #     block.attn.q = MultiModalLoRA_q(w_q_linear, i_w_a_linear_q, i_w_b_linear_q, d_w_a_linear_q, d_w_b_linear_q, e_w_a_linear_q, e_w_b_linear_q, l_w_a_linear_q, l_w_b_linear_q)
        #     w_kv_linear = block.attn.kv
        #     block.attn.kv = MultiModalLoRA_kv(w_kv_linear, i_w_a_linear_v, i_w_b_linear_v, d_w_a_linear_v, d_w_b_linear_v, e_w_a_linear_v, e_w_b_linear_v, l_w_a_linear_v, l_w_b_linear_v)
        # self._reset_lora_parameters()

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        # self.r = 16
        # self.w_a = []
        # self.w_b = []
        # for i, block in enumerate(self.block2):
        #     w_q_linear = block.attn.q
        #     dim = w_q_linear.in_features
        #     i_w_a_linear_q, i_w_b_linear_q = self._create_lora_layer(dim, self.r)
        #     i_w_a_linear_v, i_w_b_linear_v = self._create_lora_layer(dim, self.r)
        #     d_w_a_linear_q, d_w_b_linear_q = self._create_lora_layer(dim, self.r)
        #     d_w_a_linear_v, d_w_b_linear_v = self._create_lora_layer(dim, self.r)
        #     e_w_a_linear_q, e_w_b_linear_q = self._create_lora_layer(dim, self.r)
        #     e_w_a_linear_v, e_w_b_linear_v = self._create_lora_layer(dim, self.r)
        #     l_w_a_linear_q, l_w_b_linear_q = self._create_lora_layer(dim, self.r)
        #     l_w_a_linear_v, l_w_b_linear_v = self._create_lora_layer(dim, self.r)
        #     self.w_a.extend([i_w_a_linear_q, i_w_a_linear_v, d_w_a_linear_q, d_w_a_linear_v, e_w_a_linear_q, e_w_a_linear_v, l_w_a_linear_q, l_w_a_linear_v])
        #     self.w_b.extend([i_w_b_linear_q, i_w_b_linear_v, d_w_b_linear_q, d_w_b_linear_v, e_w_b_linear_q, e_w_b_linear_v, l_w_b_linear_q, l_w_b_linear_v])
        #     block.attn.q = MultiModalLoRA_q(w_q_linear, i_w_a_linear_q, i_w_b_linear_q, d_w_a_linear_q, d_w_b_linear_q, e_w_a_linear_q, e_w_b_linear_q, l_w_a_linear_q, l_w_b_linear_q)
        #     w_kv_linear = block.attn.kv
        #     block.attn.kv = MultiModalLoRA_kv(w_kv_linear, i_w_a_linear_v, i_w_b_linear_v, d_w_a_linear_v, d_w_b_linear_v, e_w_a_linear_v, e_w_b_linear_v, l_w_a_linear_v, l_w_b_linear_v)
        # self._reset_lora_parameters()

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        # self.r = 16
        # self.w_a = []
        # self.w_b = []
        # for i, block in enumerate(self.block3):
        #     w_q_linear = block.attn.q
        #     dim = w_q_linear.in_features
        #     i_w_a_linear_q, i_w_b_linear_q = self._create_lora_layer(dim, self.r)
        #     i_w_a_linear_v, i_w_b_linear_v = self._create_lora_layer(dim, self.r)
        #     d_w_a_linear_q, d_w_b_linear_q = self._create_lora_layer(dim, self.r)
        #     d_w_a_linear_v, d_w_b_linear_v = self._create_lora_layer(dim, self.r)
        #     e_w_a_linear_q, e_w_b_linear_q = self._create_lora_layer(dim, self.r)
        #     e_w_a_linear_v, e_w_b_linear_v = self._create_lora_layer(dim, self.r)
        #     l_w_a_linear_q, l_w_b_linear_q = self._create_lora_layer(dim, self.r)
        #     l_w_a_linear_v, l_w_b_linear_v = self._create_lora_layer(dim, self.r)
        #     self.w_a.extend([i_w_a_linear_q, i_w_a_linear_v, d_w_a_linear_q, d_w_a_linear_v, e_w_a_linear_q, e_w_a_linear_v, l_w_a_linear_q, l_w_a_linear_v])
        #     self.w_b.extend([i_w_b_linear_q, i_w_b_linear_v, d_w_b_linear_q, d_w_b_linear_v, e_w_b_linear_q, e_w_b_linear_v, l_w_b_linear_q, l_w_b_linear_v])
        #     block.attn.q = MultiModalLoRA_q(w_q_linear, i_w_a_linear_q, i_w_b_linear_q, d_w_a_linear_q, d_w_b_linear_q, e_w_a_linear_q, e_w_b_linear_q, l_w_a_linear_q, l_w_b_linear_q)
        #     w_kv_linear = block.attn.kv
        #     block.attn.kv = MultiModalLoRA_kv(w_kv_linear, i_w_a_linear_v, i_w_b_linear_v, d_w_a_linear_v, d_w_b_linear_v, e_w_a_linear_v, e_w_b_linear_v, l_w_a_linear_v, l_w_b_linear_v)
        # self._reset_lora_parameters()

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        # self.r = 16
        # self.w_a = []
        # self.w_b = []
        # for i, block in enumerate(self.block4):
        #     w_q_linear = block.attn.q
        #     dim = w_q_linear.in_features
        #     i_w_a_linear_q, i_w_b_linear_q = self._create_lora_layer(dim, self.r)
        #     i_w_a_linear_v, i_w_b_linear_v = self._create_lora_layer(dim, self.r)
        #     d_w_a_linear_q, d_w_b_linear_q = self._create_lora_layer(dim, self.r)
        #     d_w_a_linear_v, d_w_b_linear_v = self._create_lora_layer(dim, self.r)
        #     e_w_a_linear_q, e_w_b_linear_q = self._create_lora_layer(dim, self.r)
        #     e_w_a_linear_v, e_w_b_linear_v = self._create_lora_layer(dim, self.r)
        #     l_w_a_linear_q, l_w_b_linear_q = self._create_lora_layer(dim, self.r)
        #     l_w_a_linear_v, l_w_b_linear_v = self._create_lora_layer(dim, self.r)
        #     self.w_a.extend([i_w_a_linear_q, i_w_a_linear_v, d_w_a_linear_q, d_w_a_linear_v, e_w_a_linear_q, e_w_a_linear_v, l_w_a_linear_q, l_w_a_linear_v])
        #     self.w_b.extend([i_w_b_linear_q, i_w_b_linear_v, d_w_b_linear_q, d_w_b_linear_v, e_w_b_linear_q, e_w_b_linear_v, l_w_b_linear_q, l_w_b_linear_v])
        #     block.attn.q = MultiModalLoRA_q(w_q_linear, i_w_a_linear_q, i_w_b_linear_q, d_w_a_linear_q, d_w_b_linear_q, e_w_a_linear_q, e_w_b_linear_q, l_w_a_linear_q, l_w_b_linear_q)
        #     w_kv_linear = block.attn.kv
        #     block.attn.kv = MultiModalLoRA_kv(w_kv_linear, i_w_a_linear_v, i_w_b_linear_v, d_w_a_linear_v, d_w_b_linear_v, e_w_a_linear_v, e_w_b_linear_v, l_w_a_linear_v, l_w_b_linear_v)
        # self._reset_lora_parameters()

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)
        
    def _create_lora_layer(self, dim: int, r: int):
        w_a = nn.Linear(dim, r, bias=False)
        w_b = nn.Linear(r, dim, bias=False)
        return w_a, w_b

    def _reset_lora_parameters(self) -> None:
        for w_a in self.w_a:
            nn.init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))
        for w_b in self.w_b:
            nn.init.zeros_(w_b.weight)  

    # def load_state_dict(self, checkpoint=None, strict=True):
    #     if checkpoint is not None:
    #         new_state_dict = {}
    #         for key in checkpoint:
    #             new_key = key
    #             if 'attn.q.' in key and not 'attn.q.q.' in key:
    #                 new_key = key.replace('attn.q.', 'attn.q.q.')           
    #             if 'attn.kv.' in key and not 'attn.kv.kv.' in key:
    #                 new_key = key.replace('attn.kv.', 'attn.kv.kv.')         
    #             new_state_dict[new_key] = checkpoint[key]
            
    #         msg = super().load_state_dict(new_state_dict, strict=strict)
    #         print(msg)
            
    #     else:
    #         super().load_state_dict(path, strict=strict)
    
    @torch.no_grad()
    def _get_pos_emb(self, x: torch.Tensor):
        H_, W_ = x.shape[2:]
        y_embed = torch.arange(0, H_, 1, dtype=x.dtype, device=x.device)[None, :, None].repeat(1, 1, W_) # [1, H_, W_]
        x_embed = torch.arange(0, W_, 1, dtype=x.dtype, device=x.device)[None, None, :].repeat(1, H_, 1) # [1, H_, W_]

        if self._normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self._scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self._scale

        with amp.autocast(enabled=self._using_cuda):
            dim_t = torch.arange(self.n_features//2, dtype=x.dtype, device=x.device)
            dim_t = (self._temperature ** (2 * (torch.div(dim_t, 2, rounding_mode='trunc')) / (self.n_features//2)))

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos    
    
    @torch.no_grad()
    def _get_reference_points(self, x: torch.Tensor):
        self.n_points = 8
        B, _, H, W, = x.shape
        ref_y, ref_x = torch.meshgrid(torch.linspace(0.5/H, (H-0.5)/H, H, dtype=x.dtype, device=x.device),
                                      torch.linspace(0.5/W, (W-0.5)/W, W, dtype=x.dtype, device=x.device), indexing='ij')
        ref = torch.stack((ref_x[None, :], ref_y[None, :]), -1).expand(B*self.n_points, -1, -1, -1) # [Bxp, H, W, 2]
        ref = ref * 2 - 1
        return ref

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    '''
    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)
    '''
    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, shape, training: bool = False):
        m, b, c, h, w = shape
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            # ############################
            # if training:
            #     mb, seq_len, C = x.shape
            #     x = x.reshape(m, b, seq_len, C)
            #     mask = (torch.rand(b, seq_len) < 0.7).float().to(x.device)
            #     new_x0 = x[0]*mask.unsqueeze(-1) + x[1]*(1-mask).unsqueeze(-1)
            #     new_x1 = x[1]*mask.unsqueeze(-1) + x[0]*(1-mask).unsqueeze(-1)
            #     x = torch.stack([new_x0, new_x1, x[2]], dim=0)
            #     x = x.reshape(m*b, seq_len, C)
            # ############################
            x = blk(x, H, W)
        ############################
        # _, seq_len, C = x.shape
        # fuse_x = x.reshape(m, b, seq_len, C).permute(1, 0, 2, 3).contiguous()
        # fuse_x = fuse_x.reshape([b, m, H, W, C]).permute(0, 2, 1, 3, 4).reshape([b, m*H*W, C]).contiguous()
        # fuse_x = self.fuser1(fuse_x, (H, m*W)).reshape([b, H, m, W, C]).permute(0, 2, 1, 3, 4).reshape([b, m, seq_len, C]).contiguous()
        # fuse_x = fuse_x.permute(1, 0, 2, 3).reshape(m*b, seq_len, C).contiguous()
        # x = x + fuse_x
        ############################
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            # ############################
            # if training:
            #     mb, seq_len, C = x.shape
            #     x = x.reshape(m, b, seq_len, C)
            #     mask = (torch.rand(b, seq_len) < 0.7).float().to(x.device)
            #     new_x0 = x[0]*mask.unsqueeze(-1) + x[1]*(1-mask).unsqueeze(-1)
            #     new_x1 = x[1]*mask.unsqueeze(-1) + x[0]*(1-mask).unsqueeze(-1)
            #     x = torch.stack([new_x0, new_x1, x[2]], dim=0)
            #     x = x.reshape(m*b, seq_len, C)
            # ############################
            x = blk(x, H, W)
        ############################
        # _, seq_len, C = x.shape
        # fuse_x = x.reshape(m, b, seq_len, C).permute(1, 0, 2, 3).contiguous()
        # fuse_x = fuse_x.reshape([b, m, H, W, C]).permute(0, 2, 1, 3, 4).reshape([b, m*H*W, C]).contiguous()
        # fuse_x = self.fuser2(fuse_x, (H, m*W)).reshape([b, H, m, W, C]).permute(0, 2, 1, 3, 4).reshape([b, m, seq_len, C]).contiguous()
        # fuse_x = fuse_x.permute(1, 0, 2, 3).reshape(m*b, seq_len, C).contiguous()
        # x = x + fuse_x
        ############################        
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            # ############################
            # if training:
            #     mb, seq_len, C = x.shape
            #     x = x.reshape(m, b, seq_len, C)
            #     mask = (torch.rand(b, seq_len) < 0.7).float().to(x.device)
            #     new_x0 = x[0]*mask.unsqueeze(-1) + x[1]*(1-mask).unsqueeze(-1)
            #     new_x1 = x[1]*mask.unsqueeze(-1) + x[0]*(1-mask).unsqueeze(-1)
            #     x = torch.stack([new_x0, new_x1, x[2]], dim=0)
            #     x = x.reshape(m*b, seq_len, C)
            # ############################
            x = blk(x, H, W)
        ############################
        # _, seq_len, C = x.shape
        # fuse_x = x.reshape(m, b, seq_len, C).permute(1, 0, 2, 3).contiguous()
        # fuse_x = fuse_x.reshape([b, m, H, W, C]).permute(0, 2, 1, 3, 4).reshape([b, m*H*W, C]).contiguous()
        # fuse_x = self.fuser3(fuse_x, (H, m*W)).reshape([b, H, m, W, C]).permute(0, 2, 1, 3, 4).reshape([b, m, seq_len, C]).contiguous()
        # fuse_x = fuse_x.permute(1, 0, 2, 3).reshape(m*b, seq_len, C).contiguous()
        # x = x + fuse_x
        ############################        
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            # ############################
            # if training:
            #     mb, seq_len, C = x.shape
            #     x = x.reshape(m, b, seq_len, C)
            #     mask = (torch.rand(b, seq_len) < 0.7).float().to(x.device)
            #     new_x0 = x[0]*mask.unsqueeze(-1) + x[1]*(1-mask).unsqueeze(-1)
            #     new_x1 = x[1]*mask.unsqueeze(-1) + x[0]*(1-mask).unsqueeze(-1)
            #     x = torch.stack([new_x0, new_x1, x[2]], dim=0)
            #     x = x.reshape(m*b, seq_len, C)
            # ############################
            x = blk(x, H, W)
        ############################
        # _, seq_len, C = x.shape
        # fuse_x = x.reshape(m, b, seq_len, C).permute(1, 0, 2, 3).contiguous()
        # fuse_x = fuse_x.reshape([b, m, H, W, C]).permute(0, 2, 1, 3, 4).reshape([b, m*H*W, C]).contiguous()
        # fuse_x = self.fuser4(fuse_x, (H, m*W)).reshape([b, H, m, W, C]).permute(0, 2, 1, 3, 4).reshape([b, m, seq_len, C]).contiguous()
        # fuse_x = fuse_x.permute(1, 0, 2, 3).reshape(m*b, seq_len, C).contiguous()
        # x = x + fuse_x
        ############################        
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x, shape, training: bool = False):
        x = self.forward_features(x, shape, training=training)
        # x = self.head(x)

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class MambaVisionMixer_v9(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=8,
        d_conv=3,
        expand=1,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        n_modalities=4,
        low_rank_dim=16,
        conv_bias=True,
        bias=False,
        use_fast_path=True, 
        layer_idx=None,
        device=None,
        dtype=None,
        proj_drop=0.1,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx   
        self.n_modalities = n_modalities
        self.low_rank_dim = low_rank_dim
        # self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)    
        

        self.local_experts_in_proj = nn.ModuleList([nn.Sequential(
                                                        # nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs),
                                                        nn.Linear(self.d_model, self.low_rank_dim, bias=bias, **factory_kwargs),
                                                        nn.LayerNorm(self.low_rank_dim, **factory_kwargs),
                                                        nn.GELU(),
                                                        nn.Linear(self.low_rank_dim, self.d_inner, bias=bias, **factory_kwargs),
                                                    ) 
                                                    for _ in range(self.n_modalities)
                                                    ])
        
        self.x_proj = nn.Linear(
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        # self.local_experts_x_proj = nn.ModuleList([nn.Sequential(
        #                                                 nn.Linear(self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs),
        #                                             ) 
        #                                             for _ in range(self.n_modalities)
        #                                             ])
        
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
        # self.local_experts_dt_proj = nn.ModuleList([nn.Sequential(
        #                                                 nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs),
        #                                             )
        #                                             for _ in range(self.n_modalities)
        #                                             ])
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError        
        # if dt_init == "constant":
        #     for dt_proj in self.local_experts_dt_proj:
        #         nn.init.constant_(dt_proj[0].weight, dt_init_std)
        # elif dt_init == "random":
        #     for dt_proj in self.local_experts_dt_proj:
        #         nn.init.uniform_(dt_proj[0].weight, -dt_init_std, dt_init_std)
        # else:
        #     raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        # for dt_proj in self.local_experts_dt_proj:
        #     dt = torch.exp(
        #         torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
        #         + math.log(dt_min)
        #     ).clamp(min=dt_init_floor)
        #     inv_dt = dt + torch.log(-torch.expm1(-dt))
        #     with torch.no_grad():
        #         dt_proj[0].bias.copy_(inv_dt)
        #     dt_proj[0].bias._no_reinit = True
        
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner//2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        
        self.D = nn.Parameter(torch.ones(self.d_inner//2, device=device))
        self.D._no_weight_decay = True
        
        # self.out_proj = nn.Linear(self.d_inner//2, self.d_model, bias=bias, **factory_kwargs)
        self.local_experts_out_proj = nn.ModuleList([nn.Sequential(
                                                        nn.Linear(self.d_inner//2, self.d_model, bias=bias, **factory_kwargs),  
                                                    )
                                                    for _ in range(self.n_modalities)
                                                    ])
        
        self.proj_drop = nn.Dropout(proj_drop)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )
        
        # Extra normalization layer right before output projection
        assert RMSNormGated is not None
        self.norm = RMSNormGated(self.d_inner//2, eps=1e-5, norm_before_gate=True, **factory_kwargs)
        
        self.scale_init= 0.0001
        self.scale = nn.Parameter(torch.tensor(self.scale_init))

    def forward(self, hidden_states, modality_index = None):
        """
        hidden_states: (B, M, L, D)
        Returns: same shape as hidden_states
        """
        bs, num_m, seqlen, dim = hidden_states.shape
        assert num_m == len(modality_index)

        # xz = self.in_proj(hidden_states)
        xz = torch.stack([self.local_experts_in_proj[i](hidden_states[:, i]) for i in modality_index], dim=1).permute(0, 2, 1, 3).reshape(bs, seqlen*num_m, -1)

        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//2))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))    # [batch*num_patch, f_dim]
        # x_dbl = torch.stack([self.local_experts_x_proj[i](rearrange(x, "b d l -> (b l) d")) for i in modality_index], dim=1)#.reshape(bs, num_m*seqlen, -1)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=num_m*seqlen)
        # dt = torch.stack([rearrange(self.local_experts_dt_proj[i](dt[:,i]), "(b l) d -> b d l", l=seqlen) for i in modality_index], dim=1)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=num_m*seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=num_m*seqlen).contiguous()
        y = selective_scan_fn(x, 
                              dt, 
                              A, 
                              B, 
                              C, 
                              self.D.float(), 
                              z=None, 
                              delta_bias=self.dt_proj.bias.float(), 
                              delta_softplus=True, 
                              return_last_state=None)
        ########################################
        # if dist.get_rank() == 0: print("x", x.flatten()[:5])
        # if dist.get_rank() == 0: print("dt", dt.flatten()[:5])
        # if dist.get_rank() == 0: print("A", A.flatten()[:5])
        # if dist.get_rank() == 0: print("B", B.flatten()[:5])
        # if dist.get_rank() == 0: print("C", C.flatten()[:5])
        # if dist.get_rank() == 0: print("y", y.flatten()[:5])
        # y = y * (~torch.isnan(y))
        ########################################
        # y = x
        # y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        z = rearrange(z, "b d l -> b l d")
        y = self.norm(y, z)
        y = rearrange(y, "b (l n) d -> b n l d", l=seqlen, n=num_m)
        # out = self.out_proj(y)
        out = torch.stack([self.local_experts_out_proj[i](y[:, i]) for i in modality_index], dim=1)
        # out = self.proj_drop(out)
        return out * self.scale

class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, u, k, v):

        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.save_for_backward(w, u, k, v)
        
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = wkv_cuda.bi_wkv_forward(w, u, k, v)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        w, u, k, v = ctx.saved_tensors
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        gw, gu, gk, gv = wkv_cuda.bi_wkv_backward(w.float().contiguous(),
                          u.float().contiguous(),
                          k.float().contiguous(),
                          v.float().contiguous(),
                          gy.float().contiguous())
        if half_mode:
            return (gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            return (gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            return (gw, gu, gk, gv)

def RUN_CUDA(w, u, k, v):
    return WKV.apply(w.cuda(), u.cuda(), k.cuda(), v.cuda())

def q_shift(input, shift_pixel=1, gamma=1/4, patch_resolution=None):
    assert gamma <= 1/4
    B, N, C = input.shape
    input = input.transpose(1, 2).reshape(B, C, patch_resolution[0], patch_resolution[1])
    B, C, H, W = input.shape
    output = torch.zeros_like(input)
    output[:, 0:int(C*gamma), :, shift_pixel:W] = input[:, 0:int(C*gamma), :, 0:W-shift_pixel]
    output[:, int(C*gamma):int(C*gamma*2), :, 0:W-shift_pixel] = input[:, int(C*gamma):int(C*gamma*2), :, shift_pixel:W]
    output[:, int(C*gamma*2):int(C*gamma*3), shift_pixel:H, :] = input[:, int(C*gamma*2):int(C*gamma*3), 0:H-shift_pixel, :]
    output[:, int(C*gamma*3):int(C*gamma*4), 0:H-shift_pixel, :] = input[:, int(C*gamma*3):int(C*gamma*4), shift_pixel:H, :]
    output[:, int(C*gamma*4):, ...] = input[:, int(C*gamma*4):, ...]
    return output.flatten(2).transpose(1, 2)

class VRWKV_SpatialMix(BaseModule):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=1, init_mode='fancy',
                 key_norm=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.device = None
        attn_sz = n_embd
        self._init_weights(init_mode)
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_pixel > 0:
            self.shift_func = eval(shift_mode)
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_v = None
            self.spatial_mix_r = None

        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(attn_sz)
        else:
            self.key_norm = None
        self.output = nn.Linear(attn_sz, n_embd, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def _init_weights(self, init_mode):
        if init_mode=='fancy':
            with torch.no_grad(): # fancy init
                ratio_0_to_1 = (self.layer_id / (self.n_layer - 1)) # 0 to 1
                ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer)) # 1 to ~0
                
                # fancy time_decay
                decay_speed = torch.ones(self.n_embd)
                for h in range(self.n_embd):
                    decay_speed[h] = -5 + 8 * (h / (self.n_embd-1)) ** (0.7 + 1.3 * ratio_0_to_1)
                self.spatial_decay = nn.Parameter(decay_speed)

                # fancy time_first
                zigzag = (torch.tensor([(i+1)%3 - 1 for i in range(self.n_embd)]) * 0.5)
                self.spatial_first = nn.Parameter(torch.ones(self.n_embd) * math.log(0.3) + zigzag)
                
                # fancy time_mix
                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
                self.spatial_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))
        elif init_mode=='local':
            self.spatial_decay = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_first = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))
        elif init_mode=='global':
            self.spatial_decay = nn.Parameter(torch.zeros(self.n_embd))
            self.spatial_first = nn.Parameter(torch.zeros(self.n_embd))
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        else:
            raise NotImplementedError

    def jit_func(self, x, patch_resolution):
        # Mix x with the previous timestep to produce xk, xv, xr
        B, T, C = x.size()
        if self.shift_pixel > 0:
            xx = self.shift_func(x, self.shift_pixel, self.channel_gamma, patch_resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xv = x * self.spatial_mix_v + xx * (1 - self.spatial_mix_v)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = x
            xv = x
            xr = x

        # Use xk, xv, xr to produce k, v, r
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)

        return sr, k, v

    def forward(self, x, patch_resolution):
        B, T, C = x.size()
        self.device = x.device

        sr, k, v = self.jit_func(x, patch_resolution)
        rwkv = RUN_CUDA(self.spatial_decay / T, self.spatial_first / T, k, v)
        if self.key_norm is not None:
            rwkv = self.key_norm(rwkv)
        rwkv = sr * rwkv
        rwkv = self.output(rwkv)
        return rwkv


class VRWKV_ChannelMix(BaseModule):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=1, hidden_rate=4, init_mode='fancy',
                 key_norm=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self._init_weights(init_mode)
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_pixel > 0:
            self.shift_func = eval(shift_mode)
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_r = None

        hidden_sz = hidden_rate * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

        self.value.scale_init = 0
        self.receptance.scale_init = 0

    def _init_weights(self, init_mode):
        if init_mode == 'fancy':
            with torch.no_grad(): # fancy init of time_mix
                ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer)) # 1 to ~0
                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        elif init_mode == 'local':
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))
        elif init_mode == 'global':
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        else:
            raise NotImplementedError

    def forward(self, x, patch_resolution):
        if self.shift_pixel > 0:
            xx = self.shift_func(x, self.shift_pixel, self.channel_gamma, patch_resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = x
            xr = x

        k = self.key(xk)
        k = torch.square(torch.relu(k))
        if self.key_norm is not None:
            k = self.key_norm(k)
        kv = self.value(k)

        rkv = torch.sigmoid(self.receptance(xr)) * kv
        return rkv



class VRWKV(BaseModule):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=1, drop_path=0., hidden_rate=4,
                 init_mode='fancy', init_values=None, post_norm=False,
                 key_norm=False, with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)

        self.att = VRWKV_SpatialMix(n_embd, n_layer, layer_id, shift_mode,
                                   channel_gamma, shift_pixel, init_mode,
                                   key_norm=key_norm)
        self.ffn = VRWKV_ChannelMix(n_embd, n_layer, layer_id, shift_mode,
                                   channel_gamma, shift_pixel, hidden_rate,
                                   init_mode, key_norm=key_norm)
        self.layer_scale = (init_values is not None)
        self.post_norm = post_norm
        if self.layer_scale:
            self.gamma1 = nn.Parameter(init_values * torch.ones((n_embd)), requires_grad=True)
            self.gamma2 = nn.Parameter(init_values * torch.ones((n_embd)), requires_grad=True)
        self.with_cp = with_cp

    def forward(self, x, patch_resolution):
        if self.layer_id == 0:
            x = self.ln0(x)
        if self.post_norm:
            if self.layer_scale:
                x = x + self.drop_path(self.gamma1 * self.ln1(self.att(x, patch_resolution)))
                x = x + self.drop_path(self.gamma2 * self.ln2(self.ffn(x, patch_resolution)))
            else:
                x = x + self.drop_path(self.ln1(self.att(x, patch_resolution)))
                x = x + self.drop_path(self.ln2(self.ffn(x, patch_resolution)))
        else:
            if self.layer_scale:
                x = x + self.drop_path(self.gamma1 * self.att(self.ln1(x), patch_resolution))
                x = x + self.drop_path(self.gamma2 * self.ffn(self.ln2(x), patch_resolution))
            else:
                x = x + self.drop_path(self.att(self.ln1(x), patch_resolution))
                x = x + self.drop_path(self.ffn(self.ln2(x), patch_resolution))
        return x
    
    
class ConvGLUDownsample(nn.Module):
    
    def __init__(self, n_features=256, kernel_size=3):
        super(ConvGLUDownsample, self).__init__()

        self.n_features = n_features
        self.kernel_size = kernel_size
        _N_INPUTS = n_features * 2
        self.n_inputs = _N_INPUTS

        self.fc1 = nn.Conv2d(_N_INPUTS, _N_INPUTS, kernel_size=1, stride=1)
        self.dw_conv = nn.Conv2d(_N_INPUTS, n_features, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=n_features)
        self.sigmoid = nn.Sigmoid()
        # self.fc2 = nn.Conv2d(n_features, n_features, kernel_size=1, stride=1)

        # self._reset_parameters()

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        assert C == self.n_inputs
        x1, x2 = torch.chunk(x, chunks=2, dim=1) # [B, c, H, W]

        x = self.fc1(x) # [B, C, H, W]
        x = x.view(B, 2, self.n_features, H, W).transpose(1, 2).reshape(B, C, H, W)
        x = self.sigmoid(self.dw_conv(x)) # [B, c, H, W]
        y = x1 * x + x2 * (1 - x)
        # y = self.fc2(y) # [B, c, H, W]

        return y    
    
class DeformScaledDotAttnLayerLocal(nn.Module):
    
    def __init__(self, n_features=256, n_heads=8, n_points=8, sampling_field=7, attn_pdrop=.1, 
                 generate_offsets_with_positional_embedding=True):
        super(DeformScaledDotAttnLayerLocal, self).__init__()

        self.n_features = n_features
        self.n_heads = n_heads
        self.n_points = n_points
        _N_GROUPS = n_points*2
        assert n_features % n_heads == 0
        assert n_features % _N_GROUPS == 0
        self.n_features_per_head = n_features // n_heads

        assert sampling_field % 2 == 1
        self._SAMPLING_FIELD = sampling_field
        self._N_SAMPLERS = (self._SAMPLING_FIELD - 1) // 2
        def _gen_sampling_blocks(n_features, n_points, n_samplers):
            sampling_block_top = ConvGLUDownsample(n_features, kernel_size=3)
            sampling_block = nn.Sequential(nn.Conv2d(n_features, n_features, kernel_size=3, padding=1, groups=n_features),
                                           nn.GroupNorm(_N_GROUPS, n_features),
                                           nn.SiLU())
            sampling_block_bottom = nn.Conv2d(n_features, n_points*2, kernel_size=1)
            return nn.Sequential(*([sampling_block_top] + [sampling_block for _ in range(n_samplers-1)] + [sampling_block_bottom]))
        self.sampling_offsets = _gen_sampling_blocks(n_features, n_points, self._N_SAMPLERS)
        self.query_proj = nn.Conv2d(n_features, n_features, 1, groups=n_heads)
        self.key_proj = nn.Conv2d(n_features*n_points, n_features*n_points, 1, groups=n_heads*n_points)
        self.value_proj = nn.Conv2d(n_features*n_points, n_features*n_points, 1, groups=n_heads*n_points)
        self.output_proj = nn.Conv2d(n_features, n_features, 1)

        self._offsets_with_pos = generate_offsets_with_positional_embedding
        self.relative_position_bias_table = nn.Parameter(torch.zeros(
                    (1, self.n_heads, self._SAMPLING_FIELD, self._SAMPLING_FIELD)), requires_grad=True)
        self._bias_forward = self._table_bias_forward

        self.dropout = nn.Dropout(p=attn_pdrop)

        self._scale = math.sqrt(n_features // n_heads)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.query_proj.weight.data)
        constant_(self.query_proj.bias.data, 0.)
        xavier_uniform_(self.key_proj.weight.data)
        constant_(self.key_proj.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def _with_pos_emb(self, x: torch.Tensor, pos_emb: torch.Tensor):
        if not pos_emb is None:
            return x + pos_emb
        else:
            return x
    
    def _table_bias_forward(self, actual_offsets, bs, H, W, n_query): # generate offset bias with interpolation
        positions = actual_offsets.view(bs, self.n_points, n_query, 2).transpose(1, 2).reshape(bs*n_query, 1, self.n_points, 2) # [bsxp, H, W, 2] -> [bsxHxW, 1, p, 2]
        sampled_bias = F.grid_sample(input=self.relative_position_bias_table.expand(bs*n_query, -1, -1, -1), 
                                     grid=positions, mode='bilinear', align_corners=False) # [bsxHxW, h, 1, p]
        attn_bias = sampled_bias.view(bs, n_query, self.n_heads, self.n_points).transpose(1, 2).reshape(
            bs*self.n_heads*n_query, 1, self.n_points) # [bsxhxHxW, 1, p]
        return attn_bias

    def forward(self, query: torch.Tensor, x: torch.Tensor, reference_points: torch.Tensor, positional_embeddings: Optional[torch.Tensor] = None):
        bs_q, d_q, H_q, W_q = query.shape
        bs, d_x, H, W = x.shape
        n_query = H_q*W_q
        assert bs == bs_q, "inconsistent batch size, query: %d, x: %d" %(bs_q, bs)
        assert d_q == d_x, "inconsistent channels, query: %d, x: %d" %(d_q, d_x)
        assert H_q == H and W_q == W, "inconsistent feature map size, query: (%d, %d), x: (%d, %d)" %(H_q, W_q, H, W)

        # generating offsets
        query_embed = self._with_pos_emb(query, positional_embeddings)
        if self._offsets_with_pos:
            x_embed = self._with_pos_emb(x, positional_embeddings)
            reception = torch.cat([query_embed, x_embed], dim=1) # [bs, 2xC, H, W]
        else:
            reception = torch.cat([query, x], dim=1)
        offsets = self.sampling_offsets(reception).tanh() # [bs, 2xp, H, W]; scaled to [-1, 1]

        # normalizing and truncating sampling positions
        offsets = offsets.reshape(bs*self.n_points, 2, H, W).permute(0, 2, 3, 1) # [bs, 2xp, H, W] -> [bsxp, 2, H, W] -> [bsxp, H, W, 2]
        scaler = torch.as_tensor([self._SAMPLING_FIELD/2/W, self._SAMPLING_FIELD/2/H], 
                                  dtype=offsets.dtype, device=offsets.device)[None, None, None, :] # [1, 1, 1, 2]
        offsets = offsets * scaler # scale offsets
        sampling_loc = torch.clamp(reference_points + offsets, -1, 1).view(bs, self.n_points, H, W, 2) # [bsxp, H, W, 2] -> [bs, p, H, W, 2]

        # sampling deformed features
        sampled = []
        for i in range(self.n_points):
            sampled.append(F.grid_sample(input=x, grid=sampling_loc[:, i], mode='bilinear', align_corners=False))
        sampled = torch.stack(sampled, dim=1).view(bs, self.n_points*self.n_features, H, W) # [bs, p, C, H, W] -> [bs, p*C, H, W]

        # obtain query, key, and value
        query = self.query_proj(query_embed).reshape(bs*self.n_heads, self.n_features_per_head, n_query) # [bsxh, c, HxW]
        query = query.transpose(1, 2).reshape(bs*self.n_heads*n_query, 1, self.n_features_per_head) # [bsxhxHxW, 1, c]
        key = self.key_proj(sampled).view(bs, self.n_points, self.n_heads, self.n_features_per_head, n_query) # [bs, p, h, c, HxW]
        key = key.permute(0, 2, 4, 1, 3).reshape(bs*self.n_heads*n_query, self.n_points, self.n_features_per_head) # [bsxhxHxW, p, c]
        value = self.value_proj(sampled).view(bs, self.n_points, self.n_heads, self.n_features_per_head, n_query) # [bs, p, h, c, HxW]
        value = value.permute(0, 2, 4, 1, 3).reshape(bs*self.n_heads*n_query, self.n_points, self.n_features_per_head) # [bsxhxHxW, p, c]

        # calculating scaled-dot attention
        actual_offsets = sampling_loc.view(-1, H, W, 2) - reference_points # [bsxp, H, W, 2]
        attn_bias = self._bias_forward(actual_offsets, bs, H, W, n_query) # [bsxhxHxW, 1, p]
        attn = torch.matmul(query, key.transpose(1, 2)) / self._scale + attn_bias # [bsxhxHxW, 1, p]

        attn = self.dropout(F.softmax(attn, dim=2, dtype=attn.dtype))
        out = torch.matmul(attn, value).view(bs*self.n_heads, n_query, self.n_features_per_head) # [bsxhxHxW, 1, c] -> [bsxh, H*W, c]
        out = out.transpose(1, 2).reshape(bs, self.n_features, H, W) # [bs, C, H, W]
        output = self.output_proj(out) # [bs, C, H, W]

        return output    
    

class mit_b0(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b0, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b1(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b1, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b2(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b2, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b3(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b3, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b4(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b4, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b5(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b5, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)