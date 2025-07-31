import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from semseg.models.base import BaseModel
from semseg.models.heads import SegFormerHead
from semseg.models.heads import LightHamHead
from semseg.models.heads import UPerHead
from fvcore.nn import flop_count_table, FlopCountAnalysis
from semseg.models.backbones.dino_v2 import DinoVisionTransformer
from semseg.models.backbones.MixT import mit_b0, mit_b1, mit_b2, mit_b4, DeformScaledDotAttnLayerLocal
from semseg.models.backbones.clip import CLIPTextContextEncoder, ContextDecoder, CLIPVisionTransformer
from semseg.models.backbones.fuser import modality_agnostic_fusion
from .utils import tokenize, mask_modalities, sample_modality_mask, sample_modality_mask_v2
from semseg.models.backbones.lora import NoisyModalRouter_2D
import math
from torch.cuda import amp
import random

class SegFormer(BaseModel):
    def __init__(self, backbone: str = 'DINOv2_L', text_encoder: str = 'CLIP', img_size: int = 512, patch_size: int = 14, num_classes: int = 25, modals: list = ['img', 'depth', 'event', 'lidar'], context_length=5, embed_dim=768, use_masking=True) -> None:
        super().__init__(backbone, img_size, patch_size, num_classes, modals)
        self.num_classes = num_classes
        self.channels = 512
        # self.channels = 256
        self.context_length = context_length
        self.embed_dim = embed_dim
        self.use_masking = use_masking
        # self.router = nn.ModuleList([NoisyModalRouter_2D(embed_dim, len(self.modals), bias=False) for _ in range(4)])
        self.decode_head = SegFormerHead([self.backbone.embed_dim for i in range(4)], self.channels, self.num_classes)
        # self.decode_head = SegFormerHead([64, 128, 320, 512], self.channels, self.num_classes)
        # self.auxiliary_head = nn.ModuleList([nn.Conv2d(self.backbone.embed_dim, self.num_classes, 1) for _ in range(4)])
        self.apply(self._init_weights)

    def forward(self, x: list) -> list:
        if self.training:
            sample_id = x[-1] if isinstance(x[-1], torch.Tensor) else None
            x = torch.stack(x[:-1]).float()
        else:
            x = torch.stack(x).float()
        # x = torch.stack(x).float()
        m, b, c, h, w = x.shape
        x = x.reshape(m*b, c, h, w)
        
        if self.use_masking:
            if self.training:
                if random.random() < 0.5:
                    modality_mask = torch.ones((m, b)).to(x.device)
                else:
                    while True:
                        modality_mask = torch.bernoulli(torch.full((m, b), 0.5, device=x.device))
                        if torch.all(modality_mask.sum(0) > 0):
                            break
            else:
                modality_mask = torch.ones((m, b)).to(x.device)
                
        else:
            modality_mask = torch.ones((m, b)).to(x.device)
            
        # if self.training:
        #     # missing_mask = sample_id < 150
        #     # modality_mask[0, missing_mask] = 0 
        #     # missing_mask = sample_id > 150
        #     # modality_mask[1, missing_mask] = 0
        #     modality_mask[3, :] = 0 
        # modality_mask = torch.ones((m, b)).to(x.device)
        # modality_mask[0, :] = 1
        # modality_mask[1, :] = 1
        # modality_mask[2, :] = 0

        y, visual_embedding = self.backbone(x, modality_mask)
        y = list(y)
        modality_mask = modality_mask.long()
        all_y = []
        for i in range(4):
            y[i] = y[i].reshape(m, b, y[i].shape[1], y[i].shape[2], y[i].shape[3]) # [m, b, c, h, w]
            # modality_routing_weights = self.router[i](y[i])
            # y[i] = (y[i] * modality_routing_weights).sum(dim=0)  # [b, c, h, w]
            # y[i] = y[i].mean(dim=0)  # [b, c, h, w]
            # aux_y = self.auxiliary_head[i](y[i])  # [b, num_classes, h, w]
            # all_y.append(F.interpolate(aux_y, size=(h, w), mode='bilinear', align_corners=False))
            
            # using masking 
            mask = modality_mask.view(m, b, 1, 1, 1)  # [m, b, 1, 1, 1]
            y_masked = y[i] * mask
            y_sum = y_masked.sum(dim=0) #[b, c, h, w]
            valid_count = mask.sum(dim=0) #[b, 1, 1, 1]
            y[i] = y_sum / (valid_count + 1e-6) #[b, c, h, w]
            
            
            # mask = modality_mask.view(m, b, 1, 1, 1)  # [m, b, 1, 1, 1]
            # y[i] = y[i] * mask
            # y[i] = y[i].sum(dim=0) / mask.sum(dim=0) #[b, c, h, w]            
            
            # ## using indexing 
            # per_y = []
            # for j in range(b):
            #     tmp = []
            #     for k in range(m):
            #         if modality_mask[k][j] == 1:
            #             tmp.append(y[i][k, j])
            #     tmp = torch.stack(tmp, dim=0)
            #     per_y.append(tmp.mean(0))
            # y[i] = torch.stack(per_y, dim=0)
            
        y = self.decode_head(y)
        y = F.interpolate(y, size=(h, w), mode='bilinear', align_corners=False)
        all_y.append(y)
        
        # if not self.training:
        #     return y
        # else:
        #     return tuple(all_y)
        return y

    def init_pretrained(self, pretrained: str = None) -> None:
        if pretrained:
            checkpoint = torch.load(pretrained, map_location='cpu')#['model_state_dict']
            all_keys = list(checkpoint.keys())
            # interpolate position embedding
            if 'backbone.pos_embed' in checkpoint:
                pos_embed_checkpoint = checkpoint['backbone.pos_embed']
                embedding_size = pos_embed_checkpoint.shape[-1]
                num_patches = self.backbone.patch_embed.num_patches
                num_extra_tokens = self.backbone.pos_embed.shape[-2] - num_patches
                
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
                    checkpoint['backbone.pos_embed'] = new_pos_embed

                    patch_embed_proj = checkpoint['backbone.patch_embed.proj.weight']
                    patch_size = self.backbone.patch_embed.patch_size
                    checkpoint['backbone.patch_embed.proj.weight'] = torch.nn.functional.interpolate(
                        patch_embed_proj.float(), size=patch_size, mode='bicubic', align_corners=False)
            self.load_state_dict(checkpoint, strict=False)

def load_dualpath_model(model, model_file):
    extra_pretrained = None
    if isinstance(extra_pretrained, str):
        raw_state_dict_ext = torch.load(extra_pretrained, map_location=torch.device('cpu'))
        if 'state_dict' in raw_state_dict_ext.keys():
            raw_state_dict_ext = raw_state_dict_ext['state_dict']
    if isinstance(model_file, str):
        raw_state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        if 'model' in raw_state_dict.keys(): 
            raw_state_dict = raw_state_dict['model']
    else:
        raw_state_dict = model_file
    
    state_dict = {}
    for k, v in raw_state_dict.items():
        if k.find('patch_embed') >= 0:
            state_dict[k] = v
        elif k.find('block') >= 0:
            state_dict[k] = v
        elif k.find('norm') >= 0:
            state_dict[k] = v

    if isinstance(extra_pretrained, str):
        for k, v in raw_state_dict_ext.items():
            if k.find('patch_embed1.proj') >= 0:
                state_dict[k.replace('patch_embed1.proj', 'extra_downsample_layers.0.proj.module')] = v 
            if k.find('patch_embed2.proj') >= 0:
                state_dict[k.replace('patch_embed2.proj', 'extra_downsample_layers.1.proj.module')] = v 
            if k.find('patch_embed3.proj') >= 0:
                state_dict[k.replace('patch_embed3.proj', 'extra_downsample_layers.2.proj.module')] = v 
            if k.find('patch_embed4.proj') >= 0:
                state_dict[k.replace('patch_embed4.proj', 'extra_downsample_layers.3.proj.module')] = v 
            
            if k.find('patch_embed1.norm') >= 0:
                for i in range(model.num_modals):
                    state_dict[k.replace('patch_embed1.norm', 'extra_downsample_layers.0.norm.ln_{}'.format(i))] = v 
            if k.find('patch_embed2.norm') >= 0:
                for i in range(model.num_modals):
                    state_dict[k.replace('patch_embed2.norm', 'extra_downsample_layers.1.norm.ln_{}'.format(i))] = v 
            if k.find('patch_embed3.norm') >= 0:
                for i in range(model.num_modals):
                    state_dict[k.replace('patch_embed3.norm', 'extra_downsample_layers.2.norm.ln_{}'.format(i))] = v 
            if k.find('patch_embed4.norm') >= 0:
                for i in range(model.num_modals):
                    state_dict[k.replace('patch_embed4.norm', 'extra_downsample_layers.3.norm.ln_{}'.format(i))] = v 
            elif k.find('block') >= 0:
                state_dict[k.replace('block', 'extra_block')] = v
            elif k.find('norm') >= 0:
                state_dict[k.replace('norm', 'extra_norm')] = v


    msg = model.load_state_dict(state_dict, strict=False)
    del state_dict

if __name__ == '__main__':
    modals = ['img', 'depth', 'event', 'lidar']
    model = CMNeXt('CMNeXt-B2', 25, modals)
    model.init_pretrained('checkpoints/pretrained/segformer/mit_b2.pth')
    x = [torch.zeros(1, 3, 1024, 1024), torch.ones(1, 3, 1024, 1024), torch.ones(1, 3, 1024, 1024)*2, torch.ones(1, 3, 1024, 1024) *3]
    y = model(x)
    print(y.shape)
