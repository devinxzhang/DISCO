# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

from functools import partial
import math
from typing import Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.utils.checkpoint as cp
from mmengine.model import BaseModule
import torch.nn.functional as F
from .dino_layers import (
    Mlp,
    PatchEmbed,
    SwiGLUFFNFused,
    Attention,
    MemEffAttention,
    MulModalAttention,
    MultiModalMemEffAttention,
    NestedTensorBlock as Block,
    # NestedTensorBlock_a2b as Block,
)
from .reins import Reins, LoRAReins
from .lora import LoRA, MultiModalLoRA, MultiModalLoRA_v2, MultiModalLoRA_v3, MultiModalLoRA_v4, \
    MultiModalLoRA_proj, MLPFusionLoRA, NoisyTopkRouter, MLPFusionLoRA_wo_router, MultiModalHiRA, MLPFusionHiRA, MultiModalLoRA_qkv, \
    MultiModalLoRA_q, MultiModalLoRA_k, MultiModalLoRA_v
from .fuser import SpatialGatedFusion, OTAligner, modality_agnostic_fusion
from .MixT import DeformScaledDotAttnLayerLocal
from torch.cuda import amp
#######################################################
# from typing import OrderedDict, Sequence, Tuple, Union, Callable
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from .adapter_module import MambaVisionMixer_v9
#######################################################


def named_apply(
    fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x
    
    
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class DinoVisionTransformer(BaseModule):
    def __init__(
        self,
        model_name="L",
        modals=['rgb', 'depth', 'event', 'lidar'],
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=partial(Block, attn_class=MemEffAttention), #MulModalAttention, MemEffAttention
        ffn_layer="mlp",
        block_chunks=1,
        out_indices=[7, 11, 15, 23],
        init_cfg=None,
        use_lora=True,
        r=8,
        get_embeddings=True,
        output_dim=768,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
        """
        super().__init__(init_cfg)
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.out_indices = out_indices
        self.use_lora = use_lora
        self.r = r
        self.get_embeddings = get_embeddings
        self.embed_dim = embed_dim

        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_tokens, embed_dim)
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [
                x.item() for x in torch.linspace(0, drop_path_rate, depth)
            ]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append(
                    [nn.Identity()] * i + blocks_list[i : i + chunksize]
                )
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        
        self.modals = modals
        
        ###################################################################
        self.fpn_dim = embed_dim
        self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(self.fpn_dim, self.fpn_dim, kernel_size=2, stride=2),
                nn.SyncBatchNorm(self.fpn_dim),
                # nn.BatchNorm2d(self.fpn_dim),
                nn.GELU(),
                nn.ConvTranspose2d(self.fpn_dim, self.fpn_dim, kernel_size=2, stride=2))
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(self.fpn_dim, self.fpn_dim, kernel_size=2, stride=2))
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)  

        if get_embeddings:
            scale = embed_dim ** -0.5
            self.ln_head = LayerNorm(embed_dim)
            self.head = nn.Parameter(scale * torch.randn(embed_dim, output_dim))
        
        # ####################################################################
        # # shared LoRA
        # if self.use_lora:
        #     self.lora_layers = list(range(len(self.blocks)))
        #     self.w_a = []
        #     self.w_b = []

        #     for i, block in enumerate(self.blocks):
        #         if i not in self.lora_layers:
        #             continue
        #         w_qkv_linear = block.attn.qkv
        #         dim = w_qkv_linear.in_features

        #         w_a_linear_q, w_b_linear_q = self._create_lora_layer(dim, self.r)
        #         w_a_linear_v, w_b_linear_v = self._create_lora_layer(dim, self.r)

        #         self.w_a.extend([w_a_linear_q, w_a_linear_v])
        #         self.w_b.extend([w_b_linear_q, w_b_linear_v])

        #         block.attn.qkv = LoRA(
        #             w_qkv_linear,
        #             w_a_linear_q,
        #             w_b_linear_q,
        #             w_a_linear_v,
        #             w_b_linear_v,
        #         )
        #     self._reset_lora_parameters()
        # ####################################################################
        
    def _create_lora_layer(self, dim_in: int, dim_out: int, r: int):
        w_a = nn.Linear(dim_in, r, bias=False)
        w_b = nn.Linear(r, dim_out, bias=False)
        return w_a, w_b

    def _reset_lora_parameters(self) -> None:
        for w_a_q in self.w_a_q:
            nn.init.kaiming_uniform_(w_a_q.weight, a=math.sqrt(5))
        for w_b_q in self.w_b_q:
            nn.init.zeros_(w_b_q.weight)
        for w_a_k in self.w_a_k:
            nn.init.kaiming_uniform_(w_a_k.weight, a=math.sqrt(5))
        for w_b_k in self.w_b_k:
            nn.init.zeros_(w_b_k.weight)
        for w_a_v in self.w_a_v:
            nn.init.kaiming_uniform_(w_a_v.weight, a=math.sqrt(5))
        for w_b_v in self.w_b_v:
            nn.init.zeros_(w_b_v.weight)
        for w_a_fc1 in self.w_a_fc1:
            nn.init.kaiming_uniform_(w_a_fc1.weight, a=math.sqrt(5))
        for w_b_fc1 in self.w_b_fc1:
            nn.init.zeros_(w_b_fc1.weight)
        for w_a_fc2 in self.w_a_fc2:
            nn.init.kaiming_uniform_(w_a_fc2.weight, a=math.sqrt(5))
        for w_b_fc2 in self.w_b_fc2:
            nn.init.zeros_(w_b_fc2.weight)
            
            
    def add_init_qkv_lora(self):
        self.lora_layers = list(range(len(self.blocks)))

        for i, block in enumerate(self.blocks):
            if i not in self.lora_layers:
                continue
            
            ## LoRA for Attn
            self.w_a_q = []
            self.w_b_q = []
            self.w_a_k = []
            self.w_b_k = []
            self.w_a_v = []
            self.w_b_v = []
            w_qkv_linear = block.attn.qkv

            dim_in = w_qkv_linear.in_features
            i_w_a_linear_q, i_w_b_linear_q = self._create_lora_layer(dim_in, dim_in, self.r)
            i_w_a_linear_k, i_w_b_linear_k = self._create_lora_layer(dim_in, dim_in, self.r)
            i_w_a_linear_v, i_w_b_linear_v = self._create_lora_layer(dim_in, dim_in, self.r)
            
            d_w_a_linear_q, d_w_b_linear_q = self._create_lora_layer(dim_in, dim_in, self.r)
            d_w_a_linear_k, d_w_b_linear_k = self._create_lora_layer(dim_in, dim_in, self.r)
            d_w_a_linear_v, d_w_b_linear_v = self._create_lora_layer(dim_in, dim_in, self.r)
            
            e_w_a_linear_q, e_w_b_linear_q = self._create_lora_layer(dim_in, dim_in, self.r)
            e_w_a_linear_k, e_w_b_linear_k = self._create_lora_layer(dim_in, dim_in, self.r)
            e_w_a_linear_v, e_w_b_linear_v = self._create_lora_layer(dim_in, dim_in, self.r)
            
            l_w_a_linear_q, l_w_b_linear_q = self._create_lora_layer(dim_in, dim_in, self.r)
            l_w_a_linear_k, l_w_b_linear_k = self._create_lora_layer(dim_in, dim_in, self.r)
            l_w_a_linear_v, l_w_b_linear_v = self._create_lora_layer(dim_in, dim_in, self.r)

            self.w_a_q.extend([i_w_a_linear_q, d_w_a_linear_q, e_w_a_linear_q, l_w_a_linear_q])
            self.w_b_q.extend([i_w_b_linear_q, d_w_b_linear_q, e_w_b_linear_q, l_w_b_linear_q])
            self.w_a_k.extend([i_w_a_linear_k, d_w_a_linear_k, e_w_a_linear_k, l_w_a_linear_k])
            self.w_b_k.extend([i_w_b_linear_k, d_w_b_linear_k, e_w_b_linear_k, l_w_b_linear_k])
            self.w_a_v.extend([i_w_a_linear_v, d_w_a_linear_v, e_w_a_linear_v, l_w_a_linear_v])
            self.w_b_v.extend([i_w_b_linear_v, d_w_b_linear_v, e_w_b_linear_v, l_w_b_linear_v])

            block.attn.qkv = MultiModalLoRA( # MultiModalLoRA MultiModalHiRA 
                len(self.modals),
                w_qkv_linear,
                i_w_a_linear_q,
                i_w_b_linear_q,
                # i_w_a_linear_k,
                # i_w_b_linear_k,
                i_w_a_linear_v,
                i_w_b_linear_v,
                d_w_a_linear_q,
                d_w_b_linear_q,
                # d_w_a_linear_k,
                # d_w_b_linear_k,
                d_w_a_linear_v,
                d_w_b_linear_v,
                e_w_a_linear_q,
                e_w_b_linear_q,
                # e_w_a_linear_k,
                # e_w_b_linear_k,
                e_w_a_linear_v,
                e_w_b_linear_v,
                l_w_a_linear_q,
                l_w_b_linear_q,
                # l_w_a_linear_k,
                # l_w_b_linear_k,
                l_w_a_linear_v,
                l_w_b_linear_v,
            )

            ## LoRA for FFN 
            self.w_a_fc1 = []
            self.w_b_fc1 = []
            self.w_a_fc2 = []
            self.w_b_fc2 = []
            w_mlp = block.mlp
            w_fc1 = block.mlp.fc1
            w_fc2 = block.mlp.fc2

            dim_in1 = w_fc1.in_features
            dim_out1 = w_fc1.out_features
            dim_in2 = w_fc2.in_features
            dim_out2 = w_fc2.out_features
            i_w_a_linear_fc1, i_w_b_linear_fc1 = self._create_lora_layer(dim_in1, dim_out1, self.r)
            i_w_a_linear_fc2, i_w_b_linear_fc2 = self._create_lora_layer(dim_in2, dim_out2, self.r)
            
            d_w_a_linear_fc1, d_w_b_linear_fc1 = self._create_lora_layer(dim_in1, dim_out1, self.r)
            d_w_a_linear_fc2, d_w_b_linear_fc2 = self._create_lora_layer(dim_in2, dim_out2, self.r)
            
            e_w_a_linear_fc1, e_w_b_linear_fc1 = self._create_lora_layer(dim_in1, dim_out1, self.r)
            e_w_a_linear_fc2, e_w_b_linear_fc2 = self._create_lora_layer(dim_in2, dim_out2, self.r)
            
            l_w_a_linear_fc1, l_w_b_linear_fc1 = self._create_lora_layer(dim_in1, dim_out1, self.r)
            l_w_a_linear_fc2, l_w_b_linear_fc2 = self._create_lora_layer(dim_in2, dim_out2, self.r)

            self.w_a_fc1.extend([i_w_a_linear_fc1, d_w_a_linear_fc1, e_w_a_linear_fc1, l_w_a_linear_fc1])
            self.w_b_fc1.extend([i_w_b_linear_fc1, d_w_b_linear_fc1, e_w_b_linear_fc1, l_w_b_linear_fc1])
            self.w_a_fc2.extend([i_w_a_linear_fc2, d_w_a_linear_fc2, e_w_a_linear_fc2, l_w_a_linear_fc2])
            self.w_b_fc2.extend([i_w_b_linear_fc2, d_w_b_linear_fc2, e_w_b_linear_fc2, l_w_b_linear_fc2])

            block.mlp = MLPFusionLoRA( # MLPFusionHiRA MLPFusionLoRA MLPFusionLoRA_wo_router
                len(self.modals),
                w_mlp,
                i_w_a_linear_fc1,
                i_w_b_linear_fc1,
                i_w_a_linear_fc2,
                i_w_b_linear_fc2,
                d_w_a_linear_fc1,
                d_w_b_linear_fc1,
                d_w_a_linear_fc2,
                d_w_b_linear_fc2,
                e_w_a_linear_fc1,
                e_w_b_linear_fc1,
                e_w_a_linear_fc2,
                e_w_b_linear_fc2,
                l_w_a_linear_fc1,
                l_w_b_linear_fc1,
                l_w_a_linear_fc2,
                l_w_b_linear_fc2,
            )
            
            block.mlp.gate = nn.ModuleList([NoisyTopkRouter(dim_in1, len(self.modals), bias=False) for _ in range(len(self.modals))])
            self._reset_lora_parameters()

    def add_init_proj_lora(self):
        self.lora_layers = list(range(len(self.blocks)))

        for i, block in enumerate(self.blocks):
            if i not in self.lora_layers:
                continue
            self.w_a_o = []     
            self.w_b_o = []   
            w_proj_linear = block.attn.proj
            dim = w_proj_linear.in_features    
            i_w_a_linear_o, i_w_b_linear_o = self._create_lora_layer(dim, self.r)
            d_w_a_linear_o, d_w_b_linear_o = self._create_lora_layer(dim, self.r)
            e_w_a_linear_o, e_w_b_linear_o = self._create_lora_layer(dim, self.r)
            l_w_a_linear_o, l_w_b_linear_o = self._create_lora_layer(dim, self.r)
            self.w_a_o.extend([i_w_a_linear_o, d_w_a_linear_o, e_w_a_linear_o, l_w_a_linear_o])
            self.w_b_o.extend([i_w_b_linear_o, d_w_b_linear_o, e_w_b_linear_o, l_w_b_linear_o])
            block.attn.proj = MultiModalLoRA_proj(
                len(self.modals),
                w_proj_linear,
                i_w_a_linear_o,
                i_w_b_linear_o,
                d_w_a_linear_o,
                d_w_b_linear_o,
                e_w_a_linear_o,
                e_w_b_linear_o,
                l_w_a_linear_o,
                l_w_b_linear_o,
            )
            self._reset_proj_lora_parameters()
    ###################################################################
        
    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(
                1, int(math.sqrt(N)), int(math.sqrt(N)), dim
            ).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )

        assert (
            int(w0) == patch_pos_embed.shape[-2]
            and int(h0) == patch_pos_embed.shape[-1]
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(
            previous_dtype
        )

    def load_state_dict(self, path=None, strict=True):
        if path is not None:
            checkpoint = torch.load(path, map_location='cpu')
            all_keys = list(checkpoint.keys())
            # interpolate position embedding
            if 'pos_embed' in checkpoint:
                pos_embed_checkpoint = checkpoint['pos_embed']
                embedding_size = pos_embed_checkpoint.shape[-1]
                num_patches = self.patch_embed.num_patches
                num_extra_tokens = self.pos_embed.shape[-2] - num_patches
                # height (== width) for the checkpoint position embedding
                orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
                # height (== width) for the new position embedding
                new_size = int(num_patches ** 0.5)
                # class_token and dist_token are kept unchanged
                if orig_size != new_size:
                    print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                    # only the position tokens are interpolated
                    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                    pos_tokens = torch.nn.functional.interpolate(
                        pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                    checkpoint['pos_embed'] = new_pos_embed

                    patch_embed_proj = checkpoint['patch_embed.proj.weight']
                    patch_size = self.patch_embed.patch_size
                    checkpoint['patch_embed.proj.weight'] = torch.nn.functional.interpolate(
                        patch_embed_proj.float(), size=patch_size, mode='bicubic', align_corners=False)
            
            #################################################
            # new_state_dict = {}
            # for key in checkpoint:
            #     new_key = key
            #     if 'attn.qkv.' in key and not 'attn.qkv.qkv.' in key:
            #         new_key = key.replace('attn.qkv.', 'attn.qkv.qkv.')        
            #     if 'attn.proj.' in key and not 'attn.proj.proj.' in key:
            #         new_key = key.replace('attn.proj.', 'attn.proj.proj.')  
            #     new_state_dict[new_key] = checkpoint[key]
            #################################################
            
            msg = super().load_state_dict(checkpoint, strict=strict)
            print(msg)
            
        else:
            super().load_state_dict(path, strict=strict)

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(
                masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x
            )

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        return x

    def forward_features_list(self, x_list, masks_list):
        x = [
            self.prepare_tokens_with_masks(x, masks)
            for x, masks in zip(x_list, masks_list)
        ]
        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_patchtokens": x_norm[:, 1:],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x, masks=None):
        
        B, _, h, w = x.shape
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        x = self.prepare_tokens_with_masks(x, masks)
        outs = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            # x = cp.checkpoint(blk, x)
            if idx in self.out_indices:
                outs.append(
                    x[:, 1:, :]
                    .permute(0, 2, 1)
                    .reshape(B, -1, h // self.patch_size, w // self.patch_size)
                    .contiguous()
                )    
                
        if self.get_embeddings:
            x = self.ln_head(x)
            x = x @ self.head
                
            global_embedding = x[:, :1]
            visual_embedding = x[:, 1:].reshape(B, h // self.patch_size, w // self.patch_size, -1).permute(0, 3, 1, 2) # B C H W                
                
            return outs, [global_embedding, visual_embedding]
        else:
            return outs

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = (
            range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        )
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(
            blocks_to_take
        ), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = (
            range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        )
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(
            blocks_to_take
        ), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1:] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, **kwargs):
        x = args[0]
        modality_mask = args[1]
        # update the status of the LoRA layers
        for i, block in enumerate(self.blocks):
            if i not in self.lora_layers:
                continue
            block.mlp.modality_mask = modality_mask
        
        if self.get_embeddings:
            # ret, visual_embedding = self.forward_features(*args, **kwargs)
            ret, visual_embedding = self.forward_features(x, **kwargs)
        else:
            ret = self.forward_features(*args, **kwargs)
            
        # if isinstance(ret[0], torch.Tensor):
        #     ret[0] = F.interpolate(
        #         ret[0], scale_factor=4, mode="bilinear", align_corners=False
        #     )
        #     ret[1] = F.interpolate(
        #         ret[1], scale_factor=2, mode="bilinear", align_corners=False
        #     )
        #     ret[3] = F.interpolate(
        #         ret[3], scale_factor=0.5, mode="bilinear", align_corners=False
        #     )
        # else:
        #     ret[0][0] = F.interpolate(
        #         ret[0][0], scale_factor=4, mode="bilinear", align_corners=False
        #     )
        #     ret[0][1] = F.interpolate(
        #         ret[0][1], scale_factor=2, mode="bilinear", align_corners=False
        #     )
        #     ret[0][3] = F.interpolate(
        #         ret[0][3], scale_factor=0.5, mode="bilinear", align_corners=False
        #     )
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(ret)):
            ret[i] = ops[i](ret[i])
        if self.get_embeddings:
            return ret, visual_embedding
        else:
            return ret
    
class DINOv2_S(DinoVisionTransformer):
    def __init__(self, img_size=512, patch_size=14, modals=['rgb', 'depth', 'event', 'lidar'], **kwargs):
        super(DINOv2_S, self).__init__(
            img_size=img_size, patch_size=patch_size, modals=modals, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, out_indices=[3, 5, 7, 11], \
            init_values=1e-05, block_chunks=0, **kwargs)
        
class DINOv2_B(DinoVisionTransformer):
    def __init__(self, img_size=512, patch_size=14, modals=['rgb', 'depth', 'event', 'lidar'], **kwargs):
        super(DINOv2_B, self).__init__(
            img_size=img_size, patch_size=patch_size, modals=modals, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, out_indices=[3, 5, 7, 11], \
            init_values=1e-05, block_chunks=0, **kwargs)
        
class DINOv2_L(DinoVisionTransformer):
    def __init__(self, img_size=512, patch_size=14, modals=['rgb', 'depth', 'event', 'lidar'], **kwargs):
        super(DINOv2_L, self).__init__(
            img_size=img_size, patch_size=patch_size, modals=modals, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, out_indices=[7, 11, 15, 23], \
            init_values=1e-05, block_chunks=0, **kwargs)
        
# [2, 5, 8, 11]
# [5, 11, 17, 23]