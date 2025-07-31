# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import logging
from functools import partial

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from timm.models.layers import DropPath
from torch.nn.init import normal_
import math
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat
import random
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated, LayerNorm
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined
try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None
try:
    from Dwconv.dwconv_layer import DepthwiseFunction
except:
    DepthwiseFunction = None
import selective_scan_cuda_oflex_rh

import warnings
import math
import numbers

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

try:
    from timm.models.layers import drop_path, to_2tuple, trunc_normal_
except:
    from timm.layers import drop_path, to_2tuple, trunc_normal_
    
def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points



class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., single=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features, single)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768, single=False):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.single = single

    def forward(self, x, H, W):
        if not self.single:
            B, N, C = x.shape
            n = N // 21
            x1 = x[:, 0:16 * n, :].transpose(1, 2).view(B, C, H * 2, W * 2).contiguous()
            x2 = x[:, 16 * n:20 * n, :].transpose(1, 2).view(B, C, H, W).contiguous()
            x3 = x[:, 20 * n:, :].transpose(1, 2).view(B, C, H // 2, W // 2).contiguous()
            x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
            x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
            x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
            x = torch.cat([x1, x2, x3], dim=1)
        else:
            B, N, C = x.shape
            x = x.transpose(1, 2).view(B, C, H, W).contiguous()
            x = self.dwconv(x).flatten(2).transpose(1, 2)
        return x


class Extractor(nn.Module):
    def __init__(self, embed_dim, norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_norm = norm_layer(embed_dim)
        self.feat_norm = norm_layer(embed_dim)
        self.mlp_token2feat = nn.Linear(self.embed_dim, self.embed_dim)
        self.mlp_delta_f = nn.Linear(self.embed_dim, self.embed_dim)
        nn.init.kaiming_uniform_(self.mlp_delta_f.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mlp_token2feat.weight, a=math.sqrt(5))   
        self.gamma = nn.Parameter(init_values * torch.ones((embed_dim)), requires_grad=True)     

    def forward(self, query, feat):
        attn = torch.einsum("nbc,nmc->nbm", self.query_norm(query), self.feat_norm(feat))
        attn = attn * (self.embed_dim**-0.5)
        attn = F.softmax(attn, dim=-1)
        delta_f = torch.einsum(
            "nbm,nmc->nbc",
            attn[:, :, 1:],
            self.mlp_token2feat(feat[:, 1:, :]),
        )
        delta_f = self.mlp_delta_f(delta_f + query)
        return query + self.gamma * delta_f 


class Injector(nn.Module):
    def __init__(self, embed_dim, norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_norm = norm_layer(embed_dim)
        self.feat_norm = norm_layer(embed_dim)
        self.mlp_token2feat = nn.Linear(self.embed_dim, self.embed_dim)
        self.mlp_delta_f = nn.Linear(self.embed_dim, self.embed_dim)
        nn.init.kaiming_uniform_(self.mlp_delta_f.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mlp_token2feat.weight, a=math.sqrt(5)) 
        self.gamma = nn.Parameter(init_values * torch.ones((embed_dim)), requires_grad=True)

    def forward(self, query, feat):
        attn = torch.einsum("nbc,nmc->nbm", self.query_norm(query), self.feat_norm(feat))
        attn = attn * (self.embed_dim**-0.5)
        attn = F.softmax(attn, dim=-1)
        delta_f = torch.einsum(
            "nbm,nmc->nbc",
            attn[:, :, 1:],
            self.mlp_token2feat(feat[:, 1:, :]),
        )
        delta_f = self.mlp_delta_f(delta_f + query)
        return query + self.gamma * delta_f
            

class SpatialPriorModule(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384):
        super().__init__()

        self.stem = nn.Sequential(*[
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
        ])
        self.conv2 = nn.Sequential(*[
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(2 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv3 = nn.Sequential(*[
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv4 = nn.Sequential(*[
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        # self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        c1 = self.stem(x) # b 64 256 256
        c2 = self.conv2(c1) # b 128 128 128 
        c3 = self.conv3(c2) # b 256 64 64
        c1 = self.fc1(c1) # b 1024 64 64 
        c2 = self.fc2(c2) # b 1024 32 32
        c3 = self.fc3(c3) # b 1024 16 16

        bs, dim, _, _ = c1.shape
        c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 2s
        c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 4s
        c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 8s

        return torch.cat([c1, c2, c3], dim=1)

class InteractionBlockPrompt(nn.Module):
    def __init__(self, embed_dim, norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0):
        super().__init__()
        self.extractor = Extractor(embed_dim=embed_dim, norm_layer=norm_layer, init_values=init_values)
        self.injector = Injector(embed_dim=embed_dim, norm_layer=norm_layer, init_values=init_values)

    def forward(self, x, c):
        c = self.extractor(query=c, feat=x) # update prompt
        x_injector = self.injector(query=x, feat=c) # update feat
        return c, x_injector

class StemModule(nn.Module):
    def __init__(self, inplanes=64):
        super().__init__()

        self.stem = nn.Sequential(*[
            nn.Conv2d(3, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
        ])

    def forward(self, x):
        c1 = self.stem(x) # b 64 256 256

        return c1
    
class PEM(nn.Module):
    def __init__(self, in_channels):
        super(PEM, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
    
    def forward(self, x):
        avg_pooled = self.pool(x)
        reshaped = avg_pooled.view(x.size(0), -1)
        softmaxed = F.softmax(reshaped, dim=1)
        reshaped_back = softmaxed.view(x.size(0), x.size(1), 1, 1)
        out = x * reshaped_back
        return out

class Depthwise1DConv(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, dilation, direction='horizontal'):
        super(Depthwise1DConv, self).__init__()
        if direction == 'horizontal':
            self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=(1, kernel_size), stride=stride,
                                       padding=(0, dilation * (kernel_size // 2)), dilation=(1, dilation), groups=in_channels)
        elif direction == 'vertical':
            self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=(kernel_size, 1), stride=stride,
                                       padding=(dilation * (kernel_size // 2), 0), dilation=(dilation, 1), groups=in_channels)
    
    def forward(self, x):
        return self.depthwise(x)

class HAPE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HAPE, self).__init__()
        mid_channels = in_channels // 4
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)      
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.SyncBatchNorm(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=(1, 3), stride=2, padding=(0, 1)),
            nn.SyncBatchNorm(mid_channels),
            nn.ReLU(inplace=True),
            # PEM(mid_channels),
        )
        
        self.branch2 = nn.Sequential(
            Depthwise1DConv(mid_channels, kernel_size=3, stride=1, dilation=2, direction='horizontal'),
            nn.SyncBatchNorm(mid_channels),
            nn.ReLU(inplace=True),
            Depthwise1DConv(mid_channels, kernel_size=3, stride=2, dilation=2, direction='vertical'),
            nn.SyncBatchNorm(mid_channels),
            nn.ReLU(inplace=True),
            # PEM(mid_channels)
        )
        
        self.branch3 = nn.Sequential(
            Depthwise1DConv(mid_channels, kernel_size=5, stride=1, dilation=2, direction='horizontal'),
            nn.SyncBatchNorm(mid_channels),
            nn.ReLU(inplace=True),
            Depthwise1DConv(mid_channels, kernel_size=5, stride=2, dilation=2, direction='vertical'),
            nn.SyncBatchNorm(mid_channels),
            nn.ReLU(inplace=True),
            # PEM(mid_channels)
        )
        
        self.branch4 = nn.Sequential(
            Depthwise1DConv(mid_channels, kernel_size=7, stride=1, dilation=2, direction='horizontal'),
            nn.SyncBatchNorm(mid_channels),
            nn.ReLU(inplace=True),
            Depthwise1DConv(mid_channels, kernel_size=7, stride=2, dilation=2, direction='vertical'),
            nn.SyncBatchNorm(mid_channels),
            nn.ReLU(inplace=True),
            # PEM(mid_channels)
        )
        
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.conv1(x)
        
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        
        out = out1 + out2 + out3 + out4
        out = self.conv2(out)
        return 
    
    
class ConvTransFusion(nn.Module):
    def __init__(self, ct, cf, cm):
        super(ConvTransFusion, self).__init__()
        
        self.conv3x3 = nn.Conv2d(ct + cf, ct, kernel_size=3, padding=1)
        self.conv1x1_1 = nn.Conv2d(cf, cf, kernel_size=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1_2 = nn.Conv2d(cf, cm, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.conv1x1_3 = nn.Conv2d(ct, cf, kernel_size=1)    
        
    def forward(self, vit_feat, conv_feat):
        combine = torch.cat([conv_feat, vit_feat], dim=1)
        
        combine = self.conv3x3(combine)
        combine = self.conv1x1_1(combine)
        combine = self.global_avg_pool(combine)
        combine = self.conv1x1_2(combine)
        combine = self.sigmoid(combine)

        return combine
    
    
class TransposeTokenReEmbedding:
    @staticmethod
    def transpose_normal_padding(x, rate):
        x = rearrange(x, "b c l -> b l c")
        B, N, C = x.shape
        value = N // rate
        if N % rate != 0:
            padding_length = (value + 1) * rate - N
            padded_x = torch.nn.functional.pad(x, (0, 0, 0, padding_length))
        else:
            padded_x = x
        x_ = rearrange(padded_x, "b (k w) d -> b (w k) d", w = rate)
        x_ = rearrange(x_, "b l c -> b c l")
        return x_

    @staticmethod
    def transpose_remove_padding(x, rate):  #length
        x = rearrange(x, "b c l -> b l c")
        x = rearrange(x, "b (w k) d -> b (k w) d", w = rate)
        # x = x[:,:length,:]
        x = rearrange(x, "b l c -> b c l")
        return x    

class MambaVisionMixer(nn.Module):
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
        conv_bias=True,
        bias=False,
        use_fast_path=True, 
        layer_idx=None,
        device=None,
        dtype=None,
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
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)    
        self.x_proj = nn.Linear(
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
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
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
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
        self.scale_init= 0.001
        self.scale = nn.Parameter(torch.tensor(self.scale_init))

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//2))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))    # [batch*num_patch, f_dim]
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
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
        
        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out * self.scale
    
    
class MambaVisionMixer_Reorder(nn.Module):
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
        conv_bias=True,
        bias=False,
        use_fast_path=True, 
        layer_idx=None,
        device=None,
        dtype=None,
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
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)    
        self.x_proj = nn.Linear(
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
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
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
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
        self.scale_init= 0.001
        self.scale = nn.Parameter(torch.tensor(self.scale_init))

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)        
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//2))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))    # [batch*num_patch, f_dim]
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
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
        
        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        
        xz_b = TransposeTokenReEmbedding.transpose_normal_padding(xz, rate=2)       # [batch, feat_dim, num_patch] 
        x_b, z_b = xz_b.chunk(2, dim=1)
        x_b = F.silu(F.conv1d(input=x_b, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//2))
        z_b = F.silu(F.conv1d(input=z_b, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//2))
        x_dbl_b = self.x_proj(rearrange(x_b, "b d l -> (b l) d"))    # [batch*num_patch, f_dim]
        dt_b, B_b, C_b = torch.split(x_dbl_b, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_b = rearrange(self.dt_proj(dt_b), "(b l) d -> b d l", l=seqlen)
        B_b = rearrange(B_b, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C_b = rearrange(C_b, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y_b = selective_scan_fn(x_b, 
                              dt_b, 
                              A, 
                              B_b, 
                              C_b, 
                              self.D.float(), 
                              z=None, 
                              delta_bias=self.dt_proj.bias.float(), 
                              delta_softplus=True, 
                              return_last_state=None)
        
        y_b = torch.cat([y_b, z_b], dim=1)
        y_b = TransposeTokenReEmbedding.transpose_remove_padding(y_b, rate=2) #, length=L
        y_b = rearrange(y_b, "b d l -> b l d")
        
        out = self.out_proj(y + y_b)
        return out * self.scale
    

class MambaVisionMixer_Reorder_v2(nn.Module):
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
        conv_bias=True,
        bias=False,
        use_fast_path=True, 
        layer_idx=None,
        device=None,
        dtype=None,
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
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)    
        self.x_proj = nn.Linear(
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
        
        self.in_proj_b = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)    
        self.x_proj_b = nn.Linear(
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
            nn.init.constant_(self.dt_proj_b.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
            nn.init.uniform_(self.dt_proj_b.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        dt_b = torch.exp(
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        inv_dt_b = dt_b + torch.log(-torch.expm1(-dt_b))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
            self.dt_proj_b.bias.copy_(inv_dt_b)
        self.dt_proj.bias._no_reinit = True
        self.dt_proj_b.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner//2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        A_b = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner//2,
        ).contiguous()
        A_log_b = torch.log(A_b)
        self.A_log_b = nn.Parameter(A_log_b)
        self.A_log_b._no_weight_decay = True
        
        self.D = nn.Parameter(torch.ones(self.d_inner//2, device=device))
        self.D._no_weight_decay = True
        self.D_b = nn.Parameter(torch.ones(self.d_inner//2, device=device))
        self.D_b._no_weight_decay = True
        
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
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
        self.scale_init= 0.001
        self.scale = nn.Parameter(torch.tensor(self.scale_init))
        
        self.conv1d_x_b = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )
        self.conv1d_z_b = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )


    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)        
        xz = rearrange(xz, "b l d -> b d l")
        
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//2))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))    # [batch*num_patch, f_dim]
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
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
        
        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        
        # rate = random.choice([2, 4, 8, 16, 32, 64, 128, 256, 512])
        rate = 2
        xz_b = TransposeTokenReEmbedding.transpose_normal_padding(xz, rate=rate)       # [batch, feat_dim, num_patch] 
        # indices = torch.randperm(xz.size(2))
        # xz_b = xz[:, :, indices]
        x_b, z_b = xz_b.chunk(2, dim=1)
        A_b = -torch.exp(self.A_log_b.float())
        x_b = F.silu(F.conv1d(input=x_b, weight=self.conv1d_x_b.weight, bias=self.conv1d_x_b.bias, padding='same', groups=self.d_inner//2))
        z_b = F.silu(F.conv1d(input=z_b, weight=self.conv1d_z_b.weight, bias=self.conv1d_z_b.bias, padding='same', groups=self.d_inner//2))
        x_dbl_b = self.x_proj_b(rearrange(x_b, "b d l -> (b l) d"))    # [batch*num_patch, f_dim]
        dt_b, B_b, C_b = torch.split(x_dbl_b, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_b = rearrange(self.dt_proj_b(dt_b), "(b l) d -> b d l", l=seqlen)
        B_b = rearrange(B_b, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C_b = rearrange(C_b, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y_b = selective_scan_fn(x_b, 
                              dt_b, 
                              A_b, 
                              B_b, 
                              C_b, 
                              self.D_b.float(), 
                              z=None, 
                              delta_bias=self.dt_proj_b.bias.float(), 
                              delta_softplus=True, 
                              return_last_state=None)
        
        y_b = torch.cat([y_b, z_b], dim=1)
        y_b = TransposeTokenReEmbedding.transpose_remove_padding(y_b, rate=rate) #, length=L
        # inverse_indices = torch.argsort(indices)
        # y_b = y_b[:, :, inverse_indices]
        y_b = rearrange(y_b, "b d l -> b l d")
        
        out = self.out_proj(y + y_b)
        return out * self.scale
    

class MambaVisionMixer_v3(nn.Module):
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
        conv_bias=True,
        bias=False,
        use_fast_path=True, 
        layer_idx=None,
        device=None,
        dtype=None,
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
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)    
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            **factory_kwargs,
        )
        self.scale_init= 0.001
        self.scale = nn.Parameter(torch.tensor(self.scale_init))

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))    # [batch*num_patch, f_dim]
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
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
        
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out * self.scale
    


class MambaVisionMixer_v4(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=8,
        d_conv=3,
        conv_init=None,
        expand=1,
        headdim=128,
        ngroups=1,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        activation="swish",
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,  # Absorb kwarg for general module
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        
        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)    
        
        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
            self.init_states._no_weight_decay = True

        self.act = nn.SiLU()            
            
        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)  
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
        self.A_log._no_weight_decay = True
                
        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        # Extra normalization layer right before output projection
        assert RMSNormGated is not None
        self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)                

        self.scale_init= 0.001
        self.scale = nn.Parameter(torch.tensor(self.scale_init))

    def forward(self, u, seq_idx=None):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        _, seqlen, _ = u.shape
        
        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state)
        initial_states=repeat(self.init_states, "... -> b ...", b=batch) if self.learnable_init_states else None
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)        
        
        z, xBC, dt = torch.split(
            zxbcdt, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
        )
        dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)
        assert self.activation in ["silu", "swish"]

        # 1D Convolution
        if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
            xBC = self.act(
                self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)
            )  # (B, L, self.d_inner + 2 * ngroups * d_state)
            xBC = xBC[:, :seqlen, :]
        else:
            xBC = causal_conv1d_fn(
                x=xBC.transpose(1, 2),
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
            ).transpose(1, 2)        
        
        # Split into 3 main branches: X, B, C
        # These correspond to V, K, Q respectively in the SSM/attention duality
        x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        y = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            dt,
            A,
            rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
            rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
            chunk_size=self.chunk_size,
            D=self.D,
            z=None,
            seq_idx=seq_idx,
            initial_states=initial_states,
            **dt_limit_kwargs,
        )
        y = rearrange(y, "b l h p -> b l (h p)")

        # Multiply "gate" branch and apply extra normalization layer
        y = self.norm(y, z)
        out = self.out_proj(y)
        
        return out * self.scale
    


class MambaVisionMixer_v5(nn.Module):
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
        # self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)    
        self.in_proj = nn.Linear(self.d_model, self.d_inner//2, bias=bias, **factory_kwargs)    
        self.x_proj = nn.Linear(
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
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
        self.out_proj = nn.Linear(self.d_inner//2, self.d_model, bias=bias, **factory_kwargs)
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
        
        self.scale_init= 0.001
        self.scale = nn.Parameter(torch.tensor(self.scale_init))

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//4))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//4))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))    # [batch*num_patch, f_dim]
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
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
        
        # y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        z = rearrange(z, "b d l -> b l d")
        y = self.norm(y, z)
        out = self.out_proj(y)
        out = self.proj_drop(out)
        return out * self.scale
    
    
# share layer but different embedding
class MambaVisionMixer_share_v1(nn.Module):
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
        conv_bias=True,
        bias=False,
        use_fast_path=True, 
        layer_idx=None,
        device=None,
        dtype=None,
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
        # shared
        self.in_proj = nn.Linear(self.d_model, self.d_inner//2, bias=bias, **factory_kwargs)    

        self.mamba_wise = nn.Sequential(*[
                                    Mamba_wise_v1(d_model=d_model, 
                                    d_state=d_state, 
                                    d_conv=d_conv,
                                    expand=expand,
                                    dt_rank=dt_rank,
                                    dt_min=dt_min,
                                    dt_max=dt_max,
                                    dt_init=dt_init,
                                    dt_scale=dt_scale,
                                    dt_init_floor=dt_init_floor,
                                    conv_bias=conv_bias,
                                    bias=bias,
                                    use_fast_path=use_fast_path, 
                                    layer_idx=layer_idx,
                                    device=device,
                                    dtype=dtype) for i in range(24)])
    
        self.out_proj = nn.Linear(self.d_inner//2, self.d_model, bias=bias, **factory_kwargs)
        self.scale_init= 0.001
        self.scale = nn.Parameter(repeat(torch.tensor(self.scale_init),            
                                         " -> b ",
                                        b = 24,
                                        ).contiguous())


    def forward(self, hidden_states, layer_id):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        y = self.mamba_wise[layer_id](xz)
        out = self.out_proj(y)
        return out * self.scale[layer_id]
    

class Mamba_wise_v1(nn.Module):
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
        conv_bias=True,
        bias=False,
        use_fast_path=True, 
        layer_idx=None,
        device=None,
        dtype=None,
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
        self.x_proj = nn.Linear(
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
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
        
    def forward(self, xz):
        _, seqlen, _ = xz.shape
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//4))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//4))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))    # [batch*num_patch, f_dim]
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
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
        
        # y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        z = rearrange(z, "b d l -> b l d")
        y = self.norm(y, z)
        return y
    
    
# share layer but different embedding
class MambaVisionMixer_share_v2(nn.Module):
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
        conv_bias=True,
        bias=False,
        use_fast_path=True, 
        layer_idx=None,
        device=None,
        dtype=None,
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
        # shared
        self.in_proj = nn.Linear(self.d_model, self.d_inner//2, bias=bias, **factory_kwargs)    
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
        # layer-wise
        self.mamba_wise = nn.Sequential(*[
                                    Mamba_wise_v2(d_model=d_model, 
                                    d_state=d_state, 
                                    d_conv=d_conv,
                                    expand=expand,
                                    dt_rank=dt_rank,
                                    dt_min=dt_min,
                                    dt_max=dt_max,
                                    dt_init=dt_init,
                                    dt_scale=dt_scale,
                                    dt_init_floor=dt_init_floor,
                                    conv_bias=conv_bias,
                                    bias=bias,
                                    use_fast_path=use_fast_path, 
                                    layer_idx=layer_idx,
                                    device=device,
                                    dtype=dtype) for i in range(24)])
        # shared
        self.out_proj = nn.Linear(self.d_inner//2, self.d_model, bias=bias, **factory_kwargs)
        self.scale_init= 0.001
        self.scale = nn.Parameter(repeat(torch.tensor(self.scale_init),            
                                         " -> b ",
                                        b = 24,
                                        ).contiguous())


    def forward(self, hidden_states, layer_id):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)        
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//4))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//4))
        y = self.mamba_wise[layer_id](x, z)
        out = self.out_proj(y)
        return out * self.scale[layer_id]
    

class Mamba_wise_v2(nn.Module):
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
        conv_bias=True,
        bias=False,
        use_fast_path=True, 
        layer_idx=None,
        device=None,
        dtype=None,
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
        self.x_proj = nn.Linear(
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
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

        # Extra normalization layer right before output projection
        assert RMSNormGated is not None
        self.norm = RMSNormGated(self.d_inner//2, eps=1e-5, norm_before_gate=True, **factory_kwargs)
        
    def forward(self, x, z):
        _, _, seqlen = x.shape
        A = -torch.exp(self.A_log.float())
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))    # [batch*num_patch, f_dim]
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
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
        # y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        z = rearrange(z, "b d l -> b l d")
        y = self.norm(y, z)
        return y


class StateFusion(nn.Module):
    def __init__(self, dim):
        super(StateFusion, self).__init__()

        self.dim = dim
        self.kernel_3   = nn.Parameter(torch.ones(dim, 1, 3, 3))
        self.kernel_3_1 = nn.Parameter(torch.ones(dim, 1, 3, 3))
        self.kernel_3_2 = nn.Parameter(torch.ones(dim, 1, 3, 3))
        self.alpha = nn.Parameter(torch.ones(3), requires_grad=True)

    @staticmethod
    def padding(input_tensor, padding):
        return torch.nn.functional.pad(input_tensor, padding, mode='replicate')

    def forward(self, h):

        if self.training:
            h1 = F.conv2d(self.padding(h, (1,1,1,1)), self.kernel_3,   padding=0, dilation=1, groups=self.dim)
            h2 = F.conv2d(self.padding(h, (3,3,3,3)), self.kernel_3_1, padding=0, dilation=3, groups=self.dim)
            h3 = F.conv2d(self.padding(h, (5,5,5,5)), self.kernel_3_2, padding=0, dilation=5, groups=self.dim)
            out = self.alpha[0]*h1 + self.alpha[1]*h2 + self.alpha[2]*h3
            return out

        else:
            if not hasattr(self, "_merge_weight"):
                self._merge_weight = torch.zeros((self.dim, 1, 11, 11), device=h.device)
                self._merge_weight[:, :, 4:7, 4:7] = self.alpha[0]*self.kernel_3

                self._merge_weight[:, :, 2:3, 2:3] = self.alpha[1]*self.kernel_3_1[:,:,0:1,0:1]
                self._merge_weight[:, :, 2:3, 5:6] = self.alpha[1]*self.kernel_3_1[:,:,0:1,1:2]
                self._merge_weight[:, :, 2:3, 8:9] = self.alpha[1]*self.kernel_3_1[:,:,0:1,2:3]
                self._merge_weight[:, :, 5:6, 2:3] = self.alpha[1]*self.kernel_3_1[:,:,1:2,0:1]
                self._merge_weight[:, :, 5:6, 5:6] += self.alpha[1]*self.kernel_3_1[:,:,1:2,1:2]
                self._merge_weight[:, :, 5:6, 8:9] = self.alpha[1]*self.kernel_3_1[:,:,1:2,2:3]
                self._merge_weight[:, :, 8:9, 2:3] = self.alpha[1]*self.kernel_3_1[:,:,2:3,0:1]
                self._merge_weight[:, :, 8:9, 5:6] = self.alpha[1]*self.kernel_3_1[:,:,2:3,1:2]
                self._merge_weight[:, :, 8:9, 8:9] = self.alpha[1]*self.kernel_3_1[:,:,2:3,2:3]

                self._merge_weight[:, :, 0:1, 0:1] = self.alpha[2]*self.kernel_3_2[:,:,0:1,0:1]
                self._merge_weight[:, :, 0:1, 5:6] = self.alpha[2]*self.kernel_3_2[:,:,0:1,1:2]
                self._merge_weight[:, :, 0:1, 10:11] = self.alpha[2]*self.kernel_3_2[:,:,0:1,2:3]
                self._merge_weight[:, :, 5:6, 0:1] = self.alpha[2]*self.kernel_3_2[:,:,1:2,0:1]
                self._merge_weight[:, :, 5:6, 5:6] += self.alpha[2]*self.kernel_3_2[:,:,1:2,1:2]
                self._merge_weight[:, :, 5:6, 10:11] = self.alpha[2]*self.kernel_3_2[:,:,1:2,2:3]
                self._merge_weight[:, :, 10:11, 0:1] = self.alpha[2]*self.kernel_3_2[:,:,2:3,0:1]
                self._merge_weight[:, :, 10:11, 5:6] = self.alpha[2]*self.kernel_3_2[:,:,2:3,1:2]
                self._merge_weight[:, :, 10:11, 10:11] = self.alpha[2]*self.kernel_3_2[:,:,2:3,2:3]

            out = DepthwiseFunction.apply(h, self._merge_weight, None, 11//2, 11//2, False)

            return out

class SelectiveScanStateFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, A, B, D=None, z=None, delta_bias=None, delta_softplus=False,
                return_last_state=False, lag=0):
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if z is not None and z.stride(-1) != 1:
            z = z.contiguous()
        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
            ctx.squeeze_B = True

        out, x, *rest = selective_scan_cuda_oflex_rh.fwd(u, delta, A, B, D, delta_bias, delta_softplus, 1, True)
        ctx.delta_softplus = delta_softplus
        ctx.has_z = z is not None
        last_state = x[:, :, -1, 1::2]  # (batch, dim, dstate)
        if not ctx.has_z:
            ctx.save_for_backward(u, delta, A, B, D, delta_bias, x)
            return out if not return_last_state else (out, last_state)
        else:
            ctx.save_for_backward(u, delta, A, B, D, z, delta_bias, x, out)
            out_z = rest[0]
            return out_z if not return_last_state else (out_z, last_state)

    @staticmethod
    def backward(ctx, dout, *args):
        if not ctx.has_z:
            u, delta, A, B, D, delta_bias, x = ctx.saved_tensors
            z = None
            out = None
        else:
            u, delta, A, B, D, z, delta_bias, x, out = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        # Here we just pass in None and dz will be allocated in the C++ code.
        du, ddelta, dA, dB, dD, ddelta_bias, *rest = selective_scan_cuda_oflex_rh.bwd(
            u, delta, A, B, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        dz = rest[0] if ctx.has_z else None
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        return (du, ddelta, dA, dB,
                dD if D is not None else None,
                dz,
                ddelta_bias if delta_bias is not None else None,
                None,
                None,
                None)

# def selective_scan_fn(u, delta, A, B, D=None, z=None, delta_bias=None, delta_softplus=False,
#                      return_last_state=False):
#     """if return_last_state is True, returns (out, last_state)
#     last_state has shape (batch, dim, dstate). Note that the gradient of the last state is
#     not considered in the backward pass.
#     """

#     return SelectiveScanStateFn.apply(u, delta, A, B, D, z, delta_bias, delta_softplus, return_last_state)

class MambaVisionMixer_v6(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=1,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs) 
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            groups=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner//2, (self.dt_rank + self.d_state*2), bias=False, **factory_kwargs)
        self.x_proj_weight = nn.Parameter(self.x_proj.weight)
        del self.x_proj

        self.dt_projs = self.dt_init(self.dt_rank, self.d_inner // 2, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
        self.dt_projs_weight = nn.Parameter(self.dt_projs.weight)
        self.dt_projs_bias = nn.Parameter(self.dt_projs.bias)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner // 2, dt_init)
        self.Ds = self.D_init(self.d_inner // 2, dt_init)

        self.selective_scan = selective_scan_fn

        self.state_fusion = StateFusion(self.d_inner // 2)

        self.out_norm = nn.LayerNorm(self.d_inner // 2)
        self.out_proj = nn.Linear(self.d_inner // 2, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None


    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, bias=True,**factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=bias, **factory_kwargs)

        if bias:
            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            dt = torch.exp(
                torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))

            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)
            # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
            dt_proj.bias._no_reinit = True

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        elif dt_init == "simple":
            with torch.no_grad():
                dt_proj.weight.copy_(0.1 * torch.randn((d_inner, dt_rank)))
                dt_proj.bias.copy_(0.1 * torch.randn((d_inner)))
                dt_proj.bias._no_reinit = True
        elif dt_init == "zero":
            with torch.no_grad():
                dt_proj.weight.copy_(0.1 * torch.rand((d_inner, dt_rank)))
                dt_proj.bias.copy_(0.1 * torch.rand((d_inner)))
                dt_proj.bias._no_reinit = True
        else:
            raise NotImplementedError

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, init, device=None):
        if init=="random" or "constant":
            # S4D real initialization
            A = repeat(
                torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=d_inner,
            ).contiguous()
            A_log = torch.log(A)
            A_log = nn.Parameter(A_log)
            A_log._no_weight_decay = True
        elif init=="simple":
            A_log = nn.Parameter(torch.randn((d_inner, d_state)))
        elif init=="zero":
            A_log = nn.Parameter(torch.zeros((d_inner, d_state)))
        else:
            raise NotImplementedError
        return A_log

    @staticmethod
    def D_init(d_inner, init="random", device=None):
        if init=="random" or "constant":
            # D "skip" parameter
            D = torch.ones(d_inner, device=device)
            D = nn.Parameter(D) 
            D._no_weight_decay = True
        elif init == "simple" or "zero":
            D = nn.Parameter(torch.ones(d_inner))
        else:
            raise NotImplementedError
        return D

    def ssm(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W

        xs = x.view(B, -1, L)
        
        x_dbl = torch.matmul(self.x_proj_weight.view(1, -1, C), xs)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=1)
        dts = torch.matmul(self.dt_projs_weight.view(1, C, -1), dts)
        
        As = -torch.exp(self.A_logs)
        Ds = self.Ds
        dts = dts.contiguous()
        dt_projs_bias = self.dt_projs_bias

        h = self.selective_scan(
            xs, dts, 
            As, Bs, None,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        )

        h = rearrange(h, "b d 1 (h w) -> b (d 1) h w", h=H, w=W)
        h = self.state_fusion(h)
        h = rearrange(h, "b d h w -> b d (h w)")
        
        y = h * Cs
        y = y + xs * Ds.view(-1, 1)

        return y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) 

        x = rearrange(x, 'b h w d -> b d h w').contiguous()
        x = self.act(F.conv2d(input=x, weight=self.conv2d.weight, bias=self.conv2d.bias, padding='same', groups=self.d_inner//2))

        y = self.ssm(x) 

        y = rearrange(y, 'b d (h w)-> b h w d', h=H, w=W)

        y = self.out_norm(y)
        y = y * F.silu(z)
        y = self.out_proj(y)
        if self.dropout is not None:
            y = self.dropout(y)
        return y


class MambaVisionMixer_v7(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=1,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner // 2, bias=bias, **factory_kwargs) 
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner // 4,
            out_channels=self.d_inner // 4,
            groups=self.d_inner // 4,
            bias=conv_bias // 4,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner//4, (self.dt_rank + self.d_state*2), bias=False, **factory_kwargs)
        self.x_proj_weight = nn.Parameter(self.x_proj.weight)
        del self.x_proj

        self.dt_projs = self.dt_init(self.dt_rank, self.d_inner // 4, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
        self.dt_projs_weight = nn.Parameter(self.dt_projs.weight)
        self.dt_projs_bias = nn.Parameter(self.dt_projs.bias)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner // 4, dt_init)
        self.Ds = self.D_init(self.d_inner // 4, dt_init)

        self.selective_scan = selective_scan_fn

        self.state_fusion = StateFusion(self.d_inner // 4)

        self.out_norm = nn.LayerNorm(self.d_inner // 4)
        self.out_proj = nn.Linear(self.d_inner // 4, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None


    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, bias=True,**factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=bias, **factory_kwargs)

        if bias:
            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            dt = torch.exp(
                torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))

            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)
            # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
            dt_proj.bias._no_reinit = True

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        elif dt_init == "simple":
            with torch.no_grad():
                dt_proj.weight.copy_(0.1 * torch.randn((d_inner, dt_rank)))
                dt_proj.bias.copy_(0.1 * torch.randn((d_inner)))
                dt_proj.bias._no_reinit = True
        elif dt_init == "zero":
            with torch.no_grad():
                dt_proj.weight.copy_(0.1 * torch.rand((d_inner, dt_rank)))
                dt_proj.bias.copy_(0.1 * torch.rand((d_inner)))
                dt_proj.bias._no_reinit = True
        else:
            raise NotImplementedError

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, init, device=None):
        if init=="random" or "constant":
            # S4D real initialization
            A = repeat(
                torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=d_inner,
            ).contiguous()
            A_log = torch.log(A)
            A_log = nn.Parameter(A_log)
            A_log._no_weight_decay = True
        elif init=="simple":
            A_log = nn.Parameter(torch.randn((d_inner, d_state)))
        elif init=="zero":
            A_log = nn.Parameter(torch.zeros((d_inner, d_state)))
        else:
            raise NotImplementedError
        return A_log

    @staticmethod
    def D_init(d_inner, init="random", device=None):
        if init=="random" or "constant":
            # D "skip" parameter
            D = torch.ones(d_inner, device=device)
            D = nn.Parameter(D) 
            D._no_weight_decay = True
        elif init == "simple" or "zero":
            D = nn.Parameter(torch.ones(d_inner))
        else:
            raise NotImplementedError
        return D

    def ssm(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W

        xs = x.view(B, -1, L)
        
        x_dbl = torch.matmul(self.x_proj_weight.view(1, -1, C), xs)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=1)
        dts = torch.matmul(self.dt_projs_weight.view(1, C, -1), dts)
        
        As = -torch.exp(self.A_logs)
        Ds = self.Ds
        dts = dts.contiguous()
        dt_projs_bias = self.dt_projs_bias

        h = self.selective_scan(
            xs, dts, 
            As, Bs, None,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        )

        h = rearrange(h, "b d 1 (h w) -> b (d 1) h w", h=H, w=W)
        h = self.state_fusion(h)
        h = rearrange(h, "b d h w -> b d (h w)")
        
        y = h * Cs
        y = y + xs * Ds.view(-1, 1)

        return y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) 

        x = rearrange(x, 'b h w d -> b d h w').contiguous()
        x = self.act(F.conv2d(input=x, weight=self.conv2d.weight, bias=self.conv2d.bias, padding='same', groups=self.d_inner//4))

        y = self.ssm(x) 

        y = rearrange(y, 'b d (h w)-> b h w d', h=H, w=W)

        y = self.out_norm(y)
        y = y * F.silu(z)
        y = self.out_proj(y)
        if self.dropout is not None:
            y = self.dropout(y)
        return y



class MambaVisionMixer_v8(nn.Module):
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
        # self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)    
        self.in_proj = nn.Linear(self.d_model, self.d_inner//2, bias=bias, **factory_kwargs)    
        self.x_proj = nn.Linear(
            self.d_inner//4, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//4, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//4, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner//4,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner//4, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner//4, self.d_model, bias=bias, **factory_kwargs)
        self.proj_drop = nn.Dropout(proj_drop)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner//4,
            out_channels=self.d_inner//4,
            bias=conv_bias//4,
            kernel_size=d_conv,
            groups=self.d_inner//4,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner//4,
            out_channels=self.d_inner//4,
            bias=conv_bias//4,
            kernel_size=d_conv,
            groups=self.d_inner//4,
            **factory_kwargs,
        )
        
        # Extra normalization layer right before output projection
        assert RMSNormGated is not None
        self.norm = RMSNormGated(self.d_inner//4, eps=1e-5, norm_before_gate=True, **factory_kwargs)
        
        self.scale_init= 0.001
        self.scale = nn.Parameter(torch.tensor(self.scale_init))

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//4))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//4))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))    # [batch*num_patch, f_dim]
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
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
        
        # y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        z = rearrange(z, "b d l -> b l d")
        y = self.norm(y, z)
        out = self.out_proj(y)
        out = self.proj_drop(out)
        return out * self.scale
    
    
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
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)    
        self.x_proj = nn.Linear(
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
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
        self.out_proj = nn.Linear(self.d_inner//2, self.d_model, bias=bias, **factory_kwargs)
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
        
        self.scale_init= 0.001
        self.scale = nn.Parameter(torch.tensor(self.scale_init))

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//2))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))    # [batch*num_patch, f_dim]
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
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
        # y = x
        # y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        z = rearrange(z, "b d l -> b l d")
        y = self.norm(y, z)
        out = self.out_proj(y)
        out = self.proj_drop(out)
        return out * self.scale
    
    
    
class MambaVisionMixer_v10(nn.Module):
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
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)    
        self.x_proj = nn.Linear(
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
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
        self.out_proj = nn.Linear(self.d_inner//2, self.d_model, bias=bias, **factory_kwargs)
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
        
        self.scale_init= 0.001
        self.scale = nn.Parameter(torch.tensor(self.scale_init))
        
        ###########################################################
        self.num_layers = 24
        self.embed_dims = 1024
        self.patch_size = 16
        self.query_dims = 256
        self.token_length = 100
        self.link_token_to_query = True
        self.use_softmax = True
        self.zero_mlp_delta_f = False
        self.create_model()

        def create_model(self):
            self.learnable_tokens = nn.Parameter(
                torch.empty([self.num_layers, self.token_length, self.embed_dims])
            )
            self.scale = nn.Parameter(torch.tensor(self.scale_init))
            self.mlp_token2feat = nn.Linear(self.embed_dims, self.embed_dims)
            self.mlp_delta_f = nn.Linear(self.embed_dims, self.embed_dims)
            val = math.sqrt(
                6.0
                / float(
                    3 * reduce(mul, (self.patch_size, self.patch_size), 1) + self.embed_dims
                )
            )
            nn.init.uniform_(self.learnable_tokens.data, -val, val)
            nn.init.kaiming_uniform_(self.mlp_delta_f.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.mlp_token2feat.weight, a=math.sqrt(5))
            self.transform = nn.Linear(self.embed_dims, self.query_dims)
            self.merge = nn.Linear(self.query_dims * 3, self.query_dims)
            if self.zero_mlp_delta_f:
                del self.scale
                self.scale = 1.0
                nn.init.zeros_(self.mlp_delta_f.weight)
                nn.init.zeros_(self.mlp_delta_f.bias)
        ###########################################################

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//2))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))    # [batch*num_patch, f_dim]
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
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
        # y = x
        # y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        z = rearrange(z, "b d l -> b l d")
        y = self.norm(y, z)
        out = self.out_proj(y)
        out = self.proj_drop(out)
        return out * self.scale
    
    
    
class MambaVisionMixer_v11(nn.Module):
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
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)    
        self.x_proj = nn.Linear(
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
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
        self.out_proj = nn.Linear(self.d_inner//2, self.d_model, bias=bias, **factory_kwargs)
        self.proj_drop = nn.Dropout(proj_drop)              
        
        self.conv2d_x = StateFusion(self.d_inner // 2)
        self.conv2d_z = StateFusion(self.d_inner // 2)

        # Extra normalization layer right before output projection
        assert RMSNormGated is not None
        self.norm = RMSNormGated(self.d_inner//2, eps=1e-5, norm_before_gate=True, **factory_kwargs)
        
        self.scale_init= 0.001
        self.scale = nn.Parameter(torch.tensor(self.scale_init))

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        B, H, W, _ = hidden_states.shape
        seqlen = H*W
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b h w d -> b d h w").contiguous()
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = self.conv2d_x(x)
        z = self.conv2d_x(z)
        x_dbl = self.x_proj(rearrange(x.view(B, -1, H*W), "b d l -> (b l) d").contiguous())    # [batch*num_patch, f_dim]
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        x = rearrange(x, "b d h w -> b d (h w)").contiguous()
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen).contiguous()
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
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
        # y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d").contiguous()
        z = rearrange(z, "b d h w -> b (h w) d").contiguous()
        y = self.norm(y, z)
        out = self.out_proj(y)
        out = self.proj_drop(out)
        out = rearrange(out, "b (h w) d -> b h w d", h=H, w=W).contiguous()
        return out * self.scale
    
    
class MambaVisionMixer_v12(nn.Module):
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
        self.in_proj = nn.Linear(self.d_model, self.d_inner//2, bias=bias, **factory_kwargs)    
        self.x_proj = nn.Linear(
            self.d_inner//4, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//4, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//4, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner//4,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner//4, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner//4, self.d_model, bias=bias, **factory_kwargs)
        self.proj_drop = nn.Dropout(proj_drop)              
        
        self.conv2d_x = StateFusion(self.d_inner // 4)
        self.conv2d_z = StateFusion(self.d_inner // 4)

        # Extra normalization layer right before output projection
        assert RMSNormGated is not None
        self.norm = RMSNormGated(self.d_inner//4, eps=1e-5, norm_before_gate=True, **factory_kwargs)
        
        self.scale_init= 0.001
        self.scale = nn.Parameter(torch.tensor(self.scale_init))

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        B, H, W, _ = hidden_states.shape
        seqlen = H*W
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b h w d -> b d h w").contiguous()
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = self.conv2d_x(x)
        z = self.conv2d_x(z)
        x_dbl = self.x_proj(rearrange(x.view(B, -1, H*W), "b d l -> (b l) d").contiguous())    # [batch*num_patch, f_dim]
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        x = rearrange(x, "b d h w -> b d (h w)").contiguous()
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen).contiguous()
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
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
        # y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d").contiguous()
        z = rearrange(z, "b d h w -> b (h w) d").contiguous()
        y = self.norm(y, z)
        out = self.out_proj(y)
        out = self.proj_drop(out)
        out = rearrange(out, "b (h w) d -> b h w d", h=H, w=W).contiguous()
        return out * self.scale
    
    
class MambaVisionMixer_v13(nn.Module):
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
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)    
        self.x_proj = nn.Linear(
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
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
        self.out_proj = nn.Linear(self.d_inner//2, self.d_model, bias=bias, **factory_kwargs)
        self.proj_drop = nn.Dropout(proj_drop)              
        
        self.conv2d_x = nn.Conv2d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            groups=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )         
        self.conv2d_z = nn.Conv2d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            groups=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )       
        
        self.state_fusion = StateFusion(self.d_inner // 2)

        # Extra normalization layer right before output projection
        assert RMSNormGated is not None
        self.norm = RMSNormGated(self.d_inner//2, eps=1e-5, norm_before_gate=True, **factory_kwargs)
        
        self.scale_init= 0.001
        self.scale = nn.Parameter(torch.tensor(self.scale_init))

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        B, H, W, _ = hidden_states.shape
        seqlen = H*W
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b h w d -> b d h w")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = self.conv2d_x(x)
        z = self.conv2d_x(z)
        x_dbl = self.x_proj(rearrange(x.view(B, -1, H*W), "b d l -> (b l) d"))    # [batch*num_patch, f_dim]
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        x = rearrange(x, "b d h w -> b d (h w)").contiguous()
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
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

        y = rearrange(y, "b d (h w) -> b d h w", h=H, w=W)
        y = self.state_fusion(y)
        y = rearrange(y, "b d h w -> b d (h w)", h=H, w=W)
        y = rearrange(y, "b d l -> b l d")
        z = rearrange(z, "b d h w -> b (h w) d")
        y = self.norm(y, z)
        out = self.out_proj(y)
        out = self.proj_drop(out)
        out = rearrange(out, "b (h w) d -> b h w d", h=H, w=W)
        return out * self.scale
    
class Attention(nn.Module):
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)


        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        assert k.shape == v.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads)

        attn = torch.einsum('bnkc,bmkc->bknm', q, k) * self.scale

        attn = attn.softmax(dim=-1)

        x = torch.einsum('bknm,bmkc->bnkc', attn, v).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
    
class DeformLayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(DeformLayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)    
    
class DeformableAttention(nn.Module):
    def __init__(self, n_heads, n_channel, n_groups, offset_range_factor=1, use_pe=True, dwc_pe=False, fixed_pe=False, q_size=(36, 72), kv_size=(36, 72), activate = "gelu"):
        super(DeformableAttention, self).__init__()
        self.q_h, self.q_w = q_size
        self.kv_h, self.kv_w = kv_size
        self.nc = n_channel
        self.n_head_channels = self.nc // n_heads
        self.n_heads = n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.offset_range_factor = offset_range_factor
        self.scale = self.n_head_channels ** -0.5
        self.use_pe = use_pe
        self.dwc_pe = dwc_pe
        self.fixed_pe = fixed_pe
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups

        if activate == "gelu":
            self.conv_offset = nn.Sequential(
                    nn.Conv2d(self.n_group_channels, self.n_group_channels, 5, 4, 2, groups=self.n_group_channels, bias=False),
                    DeformLayerNorm(self.n_group_channels, 'BiasFree'),
                    nn.GELU(),
                    nn.Conv2d(self.n_group_channels, 3, 1, 1, 0, bias=False)
            )
        elif activate == "leakyrelu":
            self.conv_offset = nn.Sequential(
                    nn.Conv2d(self.n_group_channels, self.n_group_channels, 5, 4, 2, groups=self.n_group_channels, bias=False),
                    DeformLayerNorm(self.n_group_channels, 'BiasFree'),
                    nn.LeakyReLU(0.1),
                    nn.Conv2d(self.n_group_channels, 3, 1, 1, 0, bias=False)
            )
        

        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0,bias=False
        )

        self.proj_k = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0,bias=False
        )

        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0,bias=False
        )

        self.proj_out = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0,bias=False
        )

        if self.use_pe:
            if self.dwc_pe:
                self.rpe_table = nn.Conv2d(self.nc, self.nc, 
                                           kernel_size=3, stride=1, padding=1, groups=self.nc,bias=False)
            elif self.fixed_pe:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * self.q_w, self.kv_h * self.kv_w)
                )
                trunc_normal_(self.rpe_table, std=0.01)
            else:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.kv_h * 2 - 1, self.kv_w * 2 - 1)
                )
                trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None
         
    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device), 
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device)
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key).mul_(2).sub_(1)
        ref[..., 0].div_(H_key).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1) # B * g H W 2
        
        return ref
    
    def forward(self, x):
        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device

        q = self.proj_q(x)

        q_off = rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels) 

        offset = self.conv_offset(q_off) # B * g 3 Hg Wg
        offset_1, offset_2, delta_m = offset.chunk(3, dim=1)
        offset = torch.cat((offset_1, offset_2), dim=1)
        delta_m = torch.sigmoid(delta_m)
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk # n_sample个reference points

        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1.0 / Hk, 1.0 / Wk], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor) # delta{p} = s * tanh(delta{p})

        offset = rearrange(offset, 'b p h w -> b h w p') # B * g 2 Hg Wg -> B*g Hg Wg 2
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)
        
        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).tanh()
        
        x_sampled = F.grid_sample(
            input=x.reshape(B * self.n_groups, self.n_group_channels, H, W), 
            grid=pos[..., (1, 0)], # y, x -> x, y
            mode='bilinear', align_corners=True) # B * g, Cg, Hg, Wg

        x_sampled = x_sampled * delta_m
        x_sampled = x_sampled.reshape(B, C, 1, n_sample)

        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)

        attn = torch.einsum('b c m, b c n -> b m n', q, k) # B * h, HW, N_sample
        attn = attn.mul(self.scale)
        
        if self.use_pe:
            if self.dwc_pe:
                residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(B * self.n_heads, self.n_head_channels, H * W)
            elif self.fixed_pe:
                rpe_table = self.rpe_table
                attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                attn = attn + attn_bias.reshape(B * self.n_heads, H * W, self.n_sample)
            else:
                rpe_table = self.rpe_table
                rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                
                q_grid = self._get_ref_points(H, W, B, dtype, device)
                
                displacement = (q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups, n_sample, 2).unsqueeze(1)).mul(0.5)
                
                attn_bias = F.grid_sample(
                    input=rpe_bias.reshape(B * self.n_groups, self.n_group_heads, 2 * H - 1, 2 * W - 1),
                    grid=displacement[..., (1, 0)],
                    mode='bilinear', align_corners=True
                ) # B * g, h_g, HW, Ns
                
                attn_bias = attn_bias.reshape(B * self.n_heads, H * W, n_sample)
                
                attn = attn + attn_bias

        attn = F.softmax(attn, dim=2)
        out = torch.einsum('b m n, b c n -> b c m', attn, v)
        if self.use_pe and self.dwc_pe:
            out = out + residual_lepe
        out = out.reshape(B, C, H, W)
        out = self.proj_out(out)
        return out
    
    
from mmcv.cnn import (PLUGIN_LAYERS, Conv2d, ConvModule, caffe2_xavier_init,
                      normal_init, xavier_init)
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence)
from mmcv.runner import BaseModule, ModuleList

from torch.nn.modules.utils import _pair
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention

class MlvlPointGenerator:
    """Standard points generator for multi-level (Mlvl) feature maps in 2D
    points-based detectors.

    Args:
        strides (list[int] | list[tuple[int, int]]): Strides of anchors
            in multiple feature levels in order (w, h).
        offset (float): The offset of points, the value is normalized with
            corresponding stride. Defaults to 0.5.
    """
    def __init__(self, strides, offset=0.5):
        self.strides = [_pair(stride) for stride in strides]
        self.offset = offset

    @property
    def num_levels(self):
        """int: number of feature levels that the generator will be applied"""
        return len(self.strides)

    @property
    def num_base_priors(self):
        """list[int]: The number of priors (points) at a point
        on the feature grid"""
        return [1 for _ in range(len(self.strides))]

    def _meshgrid(self, x, y, row_major=True):
        yy, xx = torch.meshgrid(y, x)
        if row_major:
            # warning .flatten() would cause error in ONNX exporting
            # have to use reshape here
            return xx.reshape(-1), yy.reshape(-1)

        else:
            return yy.reshape(-1), xx.reshape(-1)

    def grid_priors(self,
                    featmap_sizes,
                    dtype=torch.float32,
                    device='cuda',
                    with_stride=False):
        """Generate grid points of multiple feature levels.

        Args:
            featmap_sizes (list[tuple]): List of feature map sizes in
                multiple feature levels, each size arrange as
                as (h, w).
            dtype (:obj:`dtype`): Dtype of priors. Default: torch.float32.
            device (str): The device where the anchors will be put on.
            with_stride (bool): Whether to concatenate the stride to
                the last dimension of points.

        Return:
            list[torch.Tensor]: Points of  multiple feature levels.
            The sizes of each tensor should be (N, 2) when with stride is
            ``False``, where N = width * height, width and height
            are the sizes of the corresponding feature level,
            and the last dimension 2 represent (coord_x, coord_y),
            otherwise the shape should be (N, 4),
            and the last dimension 4 represent
            (coord_x, coord_y, stride_w, stride_h).
        """

        assert self.num_levels == len(featmap_sizes)
        multi_level_priors = []
        for i in range(self.num_levels):
            priors = self.single_level_grid_priors(featmap_sizes[i],
                                                   level_idx=i,
                                                   dtype=dtype,
                                                   device=device,
                                                   with_stride=with_stride)
            multi_level_priors.append(priors)
        return multi_level_priors

    def single_level_grid_priors(self,
                                 featmap_size,
                                 level_idx,
                                 dtype=torch.float32,
                                 device='cuda',
                                 with_stride=False):
        """Generate grid Points of a single level.

        Note:
            This function is usually called by method ``self.grid_priors``.

        Args:
            featmap_size (tuple[int]): Size of the feature maps, arrange as
                (h, w).
            level_idx (int): The index of corresponding feature map level.
            dtype (:obj:`dtype`): Dtype of priors. Default: torch.float32.
            device (str, optional): The device the tensor will be put on.
                Defaults to 'cuda'.
            with_stride (bool): Concatenate the stride to the last dimension
                of points.

        Return:
            Tensor: Points of single feature levels.
            The shape of tensor should be (N, 2) when with stride is
            ``False``, where N = width * height, width and height
            are the sizes of the corresponding feature level,
            and the last dimension 2 represent (coord_x, coord_y),
            otherwise the shape should be (N, 4),
            and the last dimension 4 represent
            (coord_x, coord_y, stride_w, stride_h).
        """
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]
        shift_x = (torch.arange(0, feat_w, device=device) +
                   self.offset) * stride_w
        # keep featmap_size as Tensor instead of int, so that we
        # can convert to ONNX correctly
        shift_x = shift_x.to(dtype)

        shift_y = (torch.arange(0, feat_h, device=device) +
                   self.offset) * stride_h
        # keep featmap_size as Tensor instead of int, so that we
        # can convert to ONNX correctly
        shift_y = shift_y.to(dtype)
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        if not with_stride:
            shifts = torch.stack([shift_xx, shift_yy], dim=-1)
        else:
            # use `shape[0]` instead of `len(shift_xx)` for ONNX export
            stride_w = shift_xx.new_full((shift_xx.shape[0], ),
                                         stride_w).to(dtype)
            stride_h = shift_xx.new_full((shift_yy.shape[0], ),
                                         stride_h).to(dtype)
            shifts = torch.stack([shift_xx, shift_yy, stride_w, stride_h],
                                 dim=-1)
        all_points = shifts.to(device)
        return all_points

    def valid_flags(self, featmap_sizes, pad_shape, device='cuda'):
        """Generate valid flags of points of multiple feature levels.

        Args:
            featmap_sizes (list(tuple)): List of feature map sizes in
                multiple feature levels, each size arrange as
                as (h, w).
            pad_shape (tuple(int)): The padded shape of the image,
                 arrange as (h, w).
            device (str): The device where the anchors will be put on.

        Return:
            list(torch.Tensor): Valid flags of points of multiple levels.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_flags = []
        for i in range(self.num_levels):
            point_stride = self.strides[i]
            feat_h, feat_w = featmap_sizes[i]
            h, w = pad_shape[:2]
            valid_feat_h = min(int(np.ceil(h / point_stride[1])), feat_h)
            valid_feat_w = min(int(np.ceil(w / point_stride[0])), feat_w)
            flags = self.single_level_valid_flags((feat_h, feat_w),
                                                  (valid_feat_h, valid_feat_w),
                                                  device=device)
            multi_level_flags.append(flags)
        return multi_level_flags

    def single_level_valid_flags(self,
                                 featmap_size,
                                 valid_size,
                                 device='cuda'):
        """Generate the valid flags of points of a single feature map.

        Args:
            featmap_size (tuple[int]): The size of feature maps, arrange as
                as (h, w).
            valid_size (tuple[int]): The valid size of the feature maps.
                The size arrange as as (h, w).
            device (str, optional): The device where the flags will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: The valid flags of each points in a single level \
                feature map.
        """
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        return valid

    def sparse_priors(self,
                      prior_idxs,
                      featmap_size,
                      level_idx,
                      dtype=torch.float32,
                      device='cuda'):
        """Generate sparse points according to the ``prior_idxs``.

        Args:
            prior_idxs (Tensor): The index of corresponding anchors
                in the feature map.
            featmap_size (tuple[int]): feature map size arrange as (w, h).
            level_idx (int): The level index of corresponding feature
                map.
            dtype (obj:`torch.dtype`): Date type of points. Defaults to
                ``torch.float32``.
            device (obj:`torch.device`): The device where the points is
                located.
        Returns:
            Tensor: Anchor with shape (N, 2), N should be equal to
            the length of ``prior_idxs``. And last dimension
            2 represent (coord_x, coord_y).
        """
        height, width = featmap_size
        x = (prior_idxs % width + self.offset) * self.strides[level_idx][0]
        y = ((prior_idxs // width) % height +
             self.offset) * self.strides[level_idx][1]
        prioris = torch.stack([x, y], 1).to(dtype)
        prioris = prioris.to(device)
        return prioris


class MSDeformAttn(BaseModule):
    def __init__(self,
                 in_channels=[1024, 1024],
                 strides=[16, 16],
                 feat_channels=1024,
                 out_channels=1024,
                 num_outs=3,
                 norm_cfg=dict(type='GN', num_groups=32),
                 act_cfg=dict(type='ReLU'),
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=512,
                     normalize=True),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.strides = strides
        self.num_input_levels = len(in_channels)
        self.num_encoder_levels = 2
        assert self.num_encoder_levels >= 1, \
            'num_levels in attn_cfgs must be at least one'
        input_conv_list = []
        # from top to down (low to high resolution)
        for i in range(self.num_input_levels - 1,
                       self.num_input_levels - self.num_encoder_levels - 1,
                       -1):
            input_conv = ConvModule(
                in_channels[i],
                feat_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=None,
                bias=True)
            input_conv_list.append(input_conv)
        # self.input_convs = ModuleList(input_conv_list)
        self.encoder = MultiScaleDeformableAttention(embed_dims=1024,
                                                    num_heads=8,
                                                    num_levels=2,
                                                    num_points=4,
                                                    im2col_step=64,
                                                    dropout=0.0,
                                                    batch_first=False,
                                                    norm_cfg=None,
                                                    init_cfg=None)
        self.postional_encoding = build_positional_encoding(
            positional_encoding)
        # high resolution to low resolution
        self.level_encoding = nn.Embedding(self.num_encoder_levels,
                                           feat_channels)

        # fpn-like structure
        # self.lateral_convs = ModuleList()
        # self.output_convs = ModuleList()
        # self.use_bias = norm_cfg is None
        # from top to down (low to high resolution)
        # fpn for the rest features that didn't pass in encoder
        # for i in range(self.num_input_levels - self.num_encoder_levels - 1, -1,
        #                -1):
            # lateral_conv = ConvModule(
            #     in_channels[i],
            #     feat_channels,
            #     kernel_size=1,
            #     bias=self.use_bias,
            #     norm_cfg=norm_cfg,
            #     act_cfg=None)
        # self.output_conv = ConvModule(
        #     feat_channels,
        #     feat_channels,
        #     kernel_size=1,
        #     stride=1,
        #     padding=1,
        #     bias=self.use_bias,
        #     norm_cfg=norm_cfg,
        #     act_cfg=act_cfg)
            # self.lateral_convs.append(lateral_conv)
            # self.output_convs.append(output_conv)
        self.output_conv = nn.Linear(feat_channels, feat_channels)

        self.num_outs = num_outs
        self.point_generator = MlvlPointGenerator(strides)
        
        self.scale_init= 0.001
        self.scale = nn.Parameter(torch.tensor(self.scale_init))

    def init_weights(self):
        """Initialize weights."""
        for i in range(0, self.num_encoder_levels):
            xavier_init(
                self.input_convs[i].conv,
                gain=1,
                bias=0,
                distribution='uniform')

        # for i in range(0, self.num_input_levels - self.num_encoder_levels):
        #     # caffe2_xavier_init(self.lateral_convs[i].conv, bias=0)
        #     caffe2_xavier_init(self.output_convs.conv, bias=0)

        normal_init(self.level_encoding, mean=0, std=1)
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

        # init_weights defined in MultiScaleDeformableAttention
        for layer in self.encoder.layers:
            for attn in layer.attentions:
                if isinstance(attn, MultiScaleDeformableAttention):
                    attn.init_weights()

    def forward(self, feats):
        """
        Args:
            feats (list[Tensor]): Feature maps of each level. Each has
                shape of (batch_size, c, h, w).

        Returns:
            tuple: A tuple containing the following:

            - mask_feature (Tensor): shape (batch_size, c, h, w).
            - multi_scale_features (list[Tensor]): Multi scale \
                    features, each in shape (batch_size, c, h, w).
        """
        # generate padding mask for each level, for each image
        batch_size = feats[0].shape[0]
        encoder_input_list = []
        padding_mask_list = []
        level_positional_encoding_list = []
        spatial_shapes = []
        reference_points_list = []
        for i in range(self.num_encoder_levels):
            level_idx = self.num_input_levels - i - 1
            feat = feats[level_idx]
            # feat_projected = self.input_convs[i](feat)
            feat_projected = feat
            h, w = feat.shape[-2:]

            # no padding
            padding_mask_resized = feat.new_zeros(
                (batch_size, ) + feat.shape[-2:], dtype=torch.bool)
            pos_embed = self.postional_encoding(padding_mask_resized)
            level_embed = self.level_encoding.weight[i]
            level_pos_embed = level_embed.view(1, -1, 1, 1) + pos_embed
            # (h_i * w_i, 2)
            reference_points = self.point_generator.single_level_grid_priors(
                feat.shape[-2:], level_idx, device=feat.device)
            # normalize
            factor = feat.new_tensor([[w, h]]) * self.strides[level_idx]
            reference_points = reference_points / factor

            # shape (batch_size, c, h_i, w_i) -> (h_i * w_i, batch_size, c)
            feat_projected = feat_projected.flatten(2).permute(2, 0, 1)
            level_pos_embed = level_pos_embed.flatten(2).permute(2, 0, 1)
            padding_mask_resized = padding_mask_resized.flatten(1)

            encoder_input_list.append(feat_projected)
            padding_mask_list.append(padding_mask_resized)
            level_positional_encoding_list.append(level_pos_embed)
            spatial_shapes.append(feat.shape[-2:])
            reference_points_list.append(reference_points)
        # shape (batch_size, total_num_query),
        # total_num_query=sum([., h_i * w_i,.])
        padding_masks = torch.cat(padding_mask_list, dim=1)
        # shape (total_num_query, batch_size, c)
        encoder_inputs = torch.cat(encoder_input_list, dim=0)
        level_positional_encodings = torch.cat(
            level_positional_encoding_list, dim=0)
        device = encoder_inputs.device
        # shape (num_encoder_levels, 2), from low
        # resolution to high resolution
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=device)
        # shape (0, h_0*w_0, h_0*w_0+h_1*w_1, ...)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        reference_points = torch.cat(reference_points_list, dim=0)
        reference_points = reference_points[None, :, None].repeat(
            batch_size, 1, self.num_encoder_levels, 1)
        valid_radios = reference_points.new_ones(
            (batch_size, self.num_encoder_levels, 2))
        # shape (num_total_query, batch_size, c)
        memory = self.encoder(
            query=encoder_inputs,
            key=None,
            value=None,
            query_pos=level_positional_encodings,
            key_pos=None,
            attn_masks=None,
            key_padding_mask=None,
            query_key_padding_mask=padding_masks,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_radios=valid_radios)
        # (num_total_query, batch_size, c) -> (batch_size, c, num_total_query)
        memory = memory.permute(1, 2, 0)

        # from low resolution to high resolution
        num_query_per_level = [e[0] * e[1] for e in spatial_shapes]
        outs = torch.split(memory, num_query_per_level, dim=-1)
        outs = [
            x.reshape(batch_size, -1, spatial_shapes[i][0],
                      spatial_shapes[i][1]) for i, x in enumerate(outs)
        ]

        for i in range(len(outs)):
            outs[i] = self.output_conv(outs[i].reshape(batch_size, outs[i].shape[1], -1).permute(0,2,1)) * self.scale
            # outs[i] = outs[i].reshape(batch_size, outs[i].shape[1], -1).permute(0,2,1)

        return outs