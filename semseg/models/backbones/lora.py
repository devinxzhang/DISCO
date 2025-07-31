import torch
import torch.nn as nn
from .fuser import modality_agnostic_fusion
import torch.utils.checkpoint as cp
from torch.nn import functional as F

class LoRA(nn.Module):
    """Low-Rank Adaptation for the for Query (Q), Key (Q), Value (V) matrices"""

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(self.dim)

    def forward(self, x) -> torch.Tensor:
        # Compute the original qkv
        qkv = self.qkv(x)  # Shape: (B, N, 3 * org_C)

        # Compute the new q and v components
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))

        # Add new q and v components to the original qkv tensor
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v

        return qkv
    

class MultiModalLoRA(nn.Module):
    """Low-Rank Adaptation for the for Query (Q), Key (Q), Value (V) matrices"""

    def __init__(
        self,
        num_modals: int,
        qkv: nn.Module,
        i_linear_a_q: nn.Module,
        i_linear_b_q: nn.Module,
        i_linear_a_v: nn.Module,
        i_linear_b_v: nn.Module,
        d_linear_a_q: nn.Module,
        d_linear_b_q: nn.Module,
        d_linear_a_v: nn.Module,
        d_linear_b_v: nn.Module,
        e_linear_a_q: nn.Module,
        e_linear_b_q: nn.Module,
        e_linear_a_v: nn.Module,
        e_linear_b_v: nn.Module,
        l_linear_a_q: nn.Module,
        l_linear_b_q: nn.Module,
        l_linear_a_v: nn.Module,
        l_linear_b_v: nn.Module,
    ):
        super().__init__()
        self.num_modals = num_modals
        self.qkv = qkv
        self.linear_a_q = nn.ModuleList([i_linear_a_q, d_linear_a_q, e_linear_a_q, l_linear_a_q])
        self.linear_b_q = nn.ModuleList([i_linear_b_q, d_linear_b_q, e_linear_b_q, l_linear_b_q])
        self.linear_a_v = nn.ModuleList([i_linear_a_v, d_linear_a_v, e_linear_a_v, l_linear_a_v])
        self.linear_b_v = nn.ModuleList([i_linear_b_v, d_linear_b_v, e_linear_b_v, l_linear_b_v])
        # self.linear_a_q = nn.ModuleList([i_linear_a_q, d_linear_a_q, e_linear_a_q])
        # self.linear_b_q = nn.ModuleList([i_linear_b_q, d_linear_b_q, e_linear_b_q])
        # self.linear_a_v = nn.ModuleList([i_linear_a_v, d_linear_a_v, e_linear_a_v])
        # self.linear_b_v = nn.ModuleList([i_linear_b_v, d_linear_b_v, e_linear_b_v])
        self.dim = qkv.in_features
        self.w_identity = torch.eye(self.dim)

    def forward(self, x) -> torch.Tensor:
        bs_m, n, c = x.shape
        x = x.view(self.num_modals, bs_m//self.num_modals, n, c).contiguous()
        qkv = []
        for i in range(len(x)):
            # Compute the original qkv
            qkv.append(self.qkv(x[i]))  # Shape: (B, N, 3 * org_C)

            # Compute the new q and v components
            new_q = self.linear_b_q[i](self.linear_a_q[i](x[i])) #+ self.linear_b_q[3](self.linear_a_q[3](x[i]))
            new_v = self.linear_b_v[i](self.linear_a_v[i](x[i])) #+ self.linear_b_v[3](self.linear_a_v[3](x[i]))
            # new_q = self.linear_b_q[3](self.linear_a_q[3](x[i])) 
            # new_v = self.linear_b_v[3](self.linear_a_v[3](x[i])) 

            # Add new q and v components to the original qkv tensor
            qkv[i][:, :, : self.dim] += new_q
            qkv[i][:, :, -self.dim :] += new_v
        
        qkv = torch.stack(qkv).view(bs_m, n, -1).contiguous()
        return qkv
    
    
class MultiModalLoRA_qkv(nn.Module):
    """Low-Rank Adaptation for the for Query (Q), Key (Q), Value (V) matrices"""

    def __init__(
        self,
        num_modals: int,
        qkv: nn.Module,
        i_linear_a_q: nn.Module,
        i_linear_b_q: nn.Module,
        i_linear_a_k: nn.Module,
        i_linear_b_k: nn.Module,
        i_linear_a_v: nn.Module,
        i_linear_b_v: nn.Module,
        d_linear_a_q: nn.Module,
        d_linear_b_q: nn.Module,
        d_linear_a_k: nn.Module,
        d_linear_b_k: nn.Module,
        d_linear_a_v: nn.Module,
        d_linear_b_v: nn.Module,
        e_linear_a_q: nn.Module,
        e_linear_b_q: nn.Module,
        e_linear_a_k: nn.Module,
        e_linear_b_k: nn.Module,
        e_linear_a_v: nn.Module,
        e_linear_b_v: nn.Module,
        l_linear_a_q: nn.Module,
        l_linear_b_q: nn.Module,
        l_linear_a_k: nn.Module,
        l_linear_b_k: nn.Module,
        l_linear_a_v: nn.Module,
        l_linear_b_v: nn.Module,
    ):
        super().__init__()
        self.num_modals = num_modals
        self.qkv = qkv
        self.linear_a_q = nn.ModuleList([i_linear_a_q, d_linear_a_q, e_linear_a_q, l_linear_a_q])
        self.linear_b_q = nn.ModuleList([i_linear_b_q, d_linear_b_q, e_linear_b_q, l_linear_b_q])
        self.linear_a_k = nn.ModuleList([i_linear_a_k, d_linear_a_k, e_linear_a_k, l_linear_a_k])
        self.linear_b_k = nn.ModuleList([i_linear_b_k, d_linear_b_k, e_linear_b_k, l_linear_b_k])
        self.linear_a_v = nn.ModuleList([i_linear_a_v, d_linear_a_v, e_linear_a_v, l_linear_a_v])
        self.linear_b_v = nn.ModuleList([i_linear_b_v, d_linear_b_v, e_linear_b_v, l_linear_b_v])
        self.dim = qkv.in_features
        self.w_identity = torch.eye(self.dim)

    def forward(self, x) -> torch.Tensor:
        bs_m, n, c = x.shape
        x = x.view(self.num_modals, bs_m//self.num_modals, n, c).contiguous()
        qkv = []
        for i in range(len(x)):
            # Compute the original qkv
            qkv.append(self.qkv(x[i]))  # Shape: (B, N, 3 * org_C)

            # Compute the new q and v components
            new_q = self.linear_b_q[i](self.linear_a_q[i](x[i])) 
            new_k = self.linear_b_k[i](self.linear_a_k[i](x[i]))
            new_v = self.linear_b_v[i](self.linear_a_v[i](x[i])) 
            # new_q = self.linear_b_q[3](self.linear_a_q[3](x[i])) 
            # new_v = self.linear_b_v[3](self.linear_a_v[3](x[i])) 

            # Add new q and v components to the original qkv tensor
            qkv[i][:, :, : self.dim] += new_q
            qkv[i][:, :, self.dim: -self.dim] += new_k
            qkv[i][:, :, -self.dim :] += new_v
        
        qkv = torch.stack(qkv).view(bs_m, n, -1).contiguous()
        return qkv    
    
    
class MultiModalLoRA_q(nn.Module):
    """Low-Rank Adaptation for the for Query (Q), Key (Q), Value (V) matrices"""

    def __init__(
        self,
        num_modals: int,
        qkv: nn.Module,
        i_linear_a_q: nn.Module,
        i_linear_b_q: nn.Module,
        d_linear_a_q: nn.Module,
        d_linear_b_q: nn.Module,
        e_linear_a_q: nn.Module,
        e_linear_b_q: nn.Module,
        l_linear_a_q: nn.Module,
        l_linear_b_q: nn.Module,
    ):
        super().__init__()
        self.num_modals = num_modals
        self.qkv = qkv
        self.linear_a_q = nn.ModuleList([i_linear_a_q, d_linear_a_q, e_linear_a_q, l_linear_a_q])
        self.linear_b_q = nn.ModuleList([i_linear_b_q, d_linear_b_q, e_linear_b_q, l_linear_b_q])
        self.dim = qkv.in_features
        self.w_identity = torch.eye(self.dim)

    def forward(self, x) -> torch.Tensor:
        bs_m, n, c = x.shape
        x = x.view(self.num_modals, bs_m//self.num_modals, n, c).contiguous()
        qkv = []
        for i in range(len(x)):
            # Compute the original qkv
            qkv.append(self.qkv(x[i]))  # Shape: (B, N, 3 * org_C)

            # Compute the new q and v components
            new_q = self.linear_b_q[i](self.linear_a_q[i](x[i])) 

            # Add new q and v components to the original qkv tensor
            qkv[i][:, :, : self.dim] += new_q
        
        qkv = torch.stack(qkv).view(bs_m, n, -1).contiguous()
        return qkv     
    
    
class MultiModalLoRA_k(nn.Module):
    """Low-Rank Adaptation for the for Query (Q), Key (Q), Value (V) matrices"""

    def __init__(
        self,
        num_modals: int,
        qkv: nn.Module,
        i_linear_a_k: nn.Module,
        i_linear_b_k: nn.Module,
        d_linear_a_k: nn.Module,
        d_linear_b_k: nn.Module,
        e_linear_a_k: nn.Module,
        e_linear_b_k: nn.Module,
        l_linear_a_k: nn.Module,
        l_linear_b_k: nn.Module,
    ):
        super().__init__()
        self.num_modals = num_modals
        self.qkv = qkv
        self.linear_a_k = nn.ModuleList([i_linear_a_k, d_linear_a_k, e_linear_a_k, l_linear_a_k])
        self.linear_b_k = nn.ModuleList([i_linear_b_k, d_linear_b_k, e_linear_b_k, l_linear_b_k])
        self.dim = qkv.in_features
        self.w_identity = torch.eye(self.dim)

    def forward(self, x) -> torch.Tensor:
        bs_m, n, c = x.shape
        x = x.view(self.num_modals, bs_m//self.num_modals, n, c).contiguous()
        qkv = []
        for i in range(len(x)):
            # Compute the original qkv
            qkv.append(self.qkv(x[i]))  # Shape: (B, N, 3 * org_C)

            # Compute the new q and v components
            new_k = self.linear_b_k[i](self.linear_a_k[i](x[i])) 

            # Add new q and v components to the original qkv tensor
            qkv[i][:, :, self.dim: -self.dim] += new_k
        
        qkv = torch.stack(qkv).view(bs_m, n, -1).contiguous()
        return qkv 
    
    
class MultiModalLoRA_v(nn.Module):
    """Low-Rank Adaptation for the for Query (Q), Key (Q), Value (V) matrices"""

    def __init__(
        self,
        num_modals: int,
        qkv: nn.Module,
        i_linear_a_v: nn.Module,
        i_linear_b_v: nn.Module,
        d_linear_a_v: nn.Module,
        d_linear_b_v: nn.Module,
        e_linear_a_v: nn.Module,
        e_linear_b_v: nn.Module,
        l_linear_a_v: nn.Module,
        l_linear_b_v: nn.Module,
    ):
        super().__init__()
        self.num_modals = num_modals
        self.qkv = qkv
        self.linear_a_v = nn.ModuleList([i_linear_a_v, d_linear_a_v, e_linear_a_v, l_linear_a_v])
        self.linear_b_v = nn.ModuleList([i_linear_b_v, d_linear_b_v, e_linear_b_v, l_linear_b_v])
        self.dim = qkv.in_features
        self.w_identity = torch.eye(self.dim)

    def forward(self, x) -> torch.Tensor:
        bs_m, n, c = x.shape
        x = x.view(self.num_modals, bs_m//self.num_modals, n, c).contiguous()
        qkv = []
        for i in range(len(x)):
            # Compute the original qkv
            qkv.append(self.qkv(x[i]))  # Shape: (B, N, 3 * org_C)

            # Compute the new q and v components
            new_v = self.linear_b_v[i](self.linear_a_v[i](x[i])) 

            # Add new q and v components to the original qkv tensor
            qkv[i][:, :, -self.dim :] += new_v
        
        qkv = torch.stack(qkv).view(bs_m, n, -1).contiguous()
        return qkv 
    
    
class MultiModalLoRA_CLIP(nn.Module):
    """Low-Rank Adaptation for the for Query (Q), Key (Q), Value (V) matrices"""

    def __init__(
        self,
        num_modals: int,
        q: nn.Module,
        i_linear_a_q: nn.Module,
        i_linear_b_q: nn.Module,
        d_linear_a_q: nn.Module,
        d_linear_b_q: nn.Module,
        e_linear_a_q: nn.Module,
        e_linear_b_q: nn.Module,
        l_linear_a_q: nn.Module,
        l_linear_b_q: nn.Module,
    ):
        super().__init__()
        self.num_modals = num_modals
        self.q = q
        self.linear_a_q = nn.ModuleList([i_linear_a_q, d_linear_a_q, e_linear_a_q, l_linear_a_q])
        self.linear_b_q = nn.ModuleList([i_linear_b_q, d_linear_b_q, e_linear_b_q, l_linear_b_q])
        self.dim = qkv.in_features
        self.w_identity = torch.eye(self.dim)

    def forward(self, x) -> torch.Tensor:
        bs_m, n, c = x.shape
        x = x.view(self.num_modals, bs_m//self.num_modals, n, c).contiguous()
        q = []
        for i in range(len(x)):
            # Compute the original qkv
            q.append(self.q(x[i]))  # Shape: (B, N, 3 * org_C)

            # Compute the new q and v components
            new_q = self.linear_b_q[i](self.linear_a_q[i](x[i])) 

            # Add new q and v components to the original qkv tensor
            q[i] += new_q
        
        return qkv    
    
class MultiModalHiRA(nn.Module):
    """Low-Rank Adaptation for the for Query (Q), Key (Q), Value (V) matrices"""

    def __init__(
        self,
        num_modals: int,
        qkv: nn.Module,
        i_linear_a_q: nn.Module,
        i_linear_b_q: nn.Module,
        i_linear_a_v: nn.Module,
        i_linear_b_v: nn.Module,
        d_linear_a_q: nn.Module,
        d_linear_b_q: nn.Module,
        d_linear_a_v: nn.Module,
        d_linear_b_v: nn.Module,
        e_linear_a_q: nn.Module,
        e_linear_b_q: nn.Module,
        e_linear_a_v: nn.Module,
        e_linear_b_v: nn.Module,
        l_linear_a_q: nn.Module,
        l_linear_b_q: nn.Module,
        l_linear_a_v: nn.Module,
        l_linear_b_v: nn.Module,
    ):
        super().__init__()
        self.num_modals = num_modals
        self.qkv = qkv
        self.linear_a_q = nn.ModuleList([i_linear_a_q, d_linear_a_q, e_linear_a_q, l_linear_a_q])
        self.linear_b_q = nn.ModuleList([i_linear_b_q, d_linear_b_q, e_linear_b_q, l_linear_b_q])
        self.linear_a_v = nn.ModuleList([i_linear_a_v, d_linear_a_v, e_linear_a_v, l_linear_a_v])
        self.linear_b_v = nn.ModuleList([i_linear_b_v, d_linear_b_v, e_linear_b_v, l_linear_b_v])
        self.dim = qkv.in_features
        self.w_identity = torch.eye(self.dim)

    def forward(self, x) -> torch.Tensor:
        bs_m, n, c = x.shape
        x = x.view(self.num_modals, bs_m//self.num_modals, n, c).contiguous()
        qkv = []
        for i in range(len(x)):
            # Compute the original qkv
            qkv.append(self.qkv(x[i]))  # Shape: (B, N, 3 * org_C)
            
            # Construct LoRA weight via ∆W = W₀ ⊙ (A @ B)
            lora_q = torch.matmul(
                self.linear_b_q[i].weight, self.linear_a_q[i].weight
            )  # [out, in]
            delta_q_weight = self.qkv.weight[:self.dim, :] * lora_q  # Hadamard product [out, in]
            delta_q = torch.matmul(x[i], delta_q_weight.T)
            
            lora_v = torch.matmul(
                self.linear_b_v[i].weight, self.linear_a_v[i].weight
            )
            delta_v_weight = self.qkv.weight[-self.dim:, :] * lora_v
            delta_v = torch.matmul(x[i], delta_v_weight.T)

            # Add new q and v components to the original qkv tensor
            qkv[i][:, :, : self.dim] += delta_q
            qkv[i][:, :, -self.dim :] += delta_v
        
        qkv = torch.stack(qkv).view(bs_m, n, -1).contiguous()
        return qkv
        
    
class MLPFusionLoRA(nn.Module):
    """Low-Rank Adaptation for the for Query (Q), Key (Q), Value (V) matrices"""

    def __init__(
        self,
        num_modals: int,
        mlp: nn.Module,
        i_linear_a_fc1: nn.Module,
        i_linear_b_fc1: nn.Module,
        i_linear_a_fc2: nn.Module,
        i_linear_b_fc2: nn.Module,
        d_linear_a_fc1: nn.Module,
        d_linear_b_fc1: nn.Module,
        d_linear_a_fc2: nn.Module,
        d_linear_b_fc2: nn.Module,
        e_linear_a_fc1: nn.Module,
        e_linear_b_fc1: nn.Module,
        e_linear_a_fc2: nn.Module,
        e_linear_b_fc2: nn.Module,
        l_linear_a_fc1: nn.Module,
        l_linear_b_fc1: nn.Module,
        l_linear_a_fc2: nn.Module,
        l_linear_b_fc2: nn.Module,
        gate: nn.ModuleList = None,
        modality_mask: torch.Tensor = None,
    ):
        super().__init__()
        self.num_modals = num_modals
        self.mlp = mlp
        self.linear_a_fc1 = nn.ModuleList([i_linear_a_fc1, d_linear_a_fc1, e_linear_a_fc1, l_linear_a_fc1])
        self.linear_b_fc1 = nn.ModuleList([i_linear_b_fc1, d_linear_b_fc1, e_linear_b_fc1, l_linear_b_fc1])
        self.linear_a_fc2 = nn.ModuleList([i_linear_a_fc2, d_linear_a_fc2, e_linear_a_fc2, l_linear_a_fc2])
        self.linear_b_fc2 = nn.ModuleList([i_linear_b_fc2, d_linear_b_fc2, e_linear_b_fc2, l_linear_b_fc2])
        # self.linear_a_fc1 = nn.ModuleList([i_linear_a_fc1, d_linear_a_fc1, e_linear_a_fc1])
        # self.linear_b_fc1 = nn.ModuleList([i_linear_b_fc1, d_linear_b_fc1, e_linear_b_fc1])
        # self.linear_a_fc2 = nn.ModuleList([i_linear_a_fc2, d_linear_a_fc2, e_linear_a_fc2])
        # self.linear_b_fc2 = nn.ModuleList([i_linear_b_fc2, d_linear_b_fc2, e_linear_b_fc2])
        self.gate = gate
        self.modality_mask = modality_mask

    def forward(self, x) -> torch.Tensor:
        bs_m, n, c = x.shape
        x1 = self.mlp.fc1(x).view(self.num_modals, bs_m//self.num_modals, n, -1).contiguous()
        
        x = x.view(self.num_modals, bs_m//self.num_modals, n, c).contiguous()
        routing_weights = []
        lora1 = []
        for i in range(x.shape[0]):
            routing_weights.append(F.softmax(self.gate[i](x[i].view(-1, c)), dim=1, dtype=torch.float).view(bs_m//self.num_modals, n, -1))
            lora1.append(self.linear_b_fc1[i](self.linear_a_fc1[i](x[i])))
        lora1 = torch.stack(lora1)
        routing_weights = torch.stack(routing_weights)
        
        fused_lora1 = torch.zeros_like(x1)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if self.modality_mask[i][j] == 1:
                    per_routing_weights = routing_weights[i,j,:] * self.modality_mask[:,j].unsqueeze(0)
                    per_routing_weights /= (per_routing_weights.sum(dim=-1, keepdim=True) + 1e-6)
                    fused_lora1[i][j] = (lora1[:,j,:,:] * per_routing_weights.permute(1,0)[:,:,None]).sum(0)
        x1 = x1 + fused_lora1
        x1 = x1.view(bs_m, n, -1).contiguous()

        x1 = self.mlp.act(x1)
        x1 = self.mlp.drop(x1)
        x1 = x1.view(self.num_modals, bs_m//self.num_modals, n, -1).contiguous()
        x2 = self.mlp.fc2(x1).view(self.num_modals, bs_m//self.num_modals, n, -1).contiguous()

        lora2 = []
        for i in range(x.shape[0]):
            lora2.append(self.linear_b_fc2[i](self.linear_a_fc2[i](x1[i])))
        lora2 = torch.stack(lora2)
        
        fused_lora2 = torch.zeros_like(x2)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if self.modality_mask[i][j] == 1:
                    per_routing_weights = routing_weights[i,j,:] * self.modality_mask[:,j].unsqueeze(0)
                    per_routing_weights /= (per_routing_weights.sum(dim=-1, keepdim=True) + 1e-6)
                    fused_lora2[i][j] = (lora2[:,j,:,:] * per_routing_weights.permute(1,0)[:,:,None]).sum(0)
        x2 = x2 + fused_lora2
        x2 = x2.view(bs_m, n, -1).contiguous()
        
        x2 = self.mlp.drop(x2)
        return x2
    
class MLPFusionLoRA_CLIP(nn.Module):
    """Low-Rank Adaptation for the for Query (Q), Key (Q), Value (V) matrices"""

    def __init__(
        self,
        num_modals: int,
        mlp: nn.Module,
        i_linear_a_fc1: nn.Module,
        i_linear_b_fc1: nn.Module,
        i_linear_a_fc2: nn.Module,
        i_linear_b_fc2: nn.Module,
        d_linear_a_fc1: nn.Module,
        d_linear_b_fc1: nn.Module,
        d_linear_a_fc2: nn.Module,
        d_linear_b_fc2: nn.Module,
        e_linear_a_fc1: nn.Module,
        e_linear_b_fc1: nn.Module,
        e_linear_a_fc2: nn.Module,
        e_linear_b_fc2: nn.Module,
        l_linear_a_fc1: nn.Module,
        l_linear_b_fc1: nn.Module,
        l_linear_a_fc2: nn.Module,
        l_linear_b_fc2: nn.Module,
        gate: nn.ModuleList = None,
        modality_mask: torch.Tensor = None,
    ):
        super().__init__()
        self.num_modals = num_modals
        self.mlp = mlp
        self.linear_a_fc1 = nn.ModuleList([i_linear_a_fc1, d_linear_a_fc1, e_linear_a_fc1, l_linear_a_fc1])
        self.linear_b_fc1 = nn.ModuleList([i_linear_b_fc1, d_linear_b_fc1, e_linear_b_fc1, l_linear_b_fc1])
        self.linear_a_fc2 = nn.ModuleList([i_linear_a_fc2, d_linear_a_fc2, e_linear_a_fc2, l_linear_a_fc2])
        self.linear_b_fc2 = nn.ModuleList([i_linear_b_fc2, d_linear_b_fc2, e_linear_b_fc2, l_linear_b_fc2])
        self.gate = gate
        self.modality_mask = modality_mask

    def forward(self, x) -> torch.Tensor:
        bs_m, n, c = x.shape
        x1 = self.mlp[0](x).view(self.num_modals, bs_m//self.num_modals, n, -1).contiguous()
        
        x = x.view(self.num_modals, bs_m//self.num_modals, n, c).contiguous()
        routing_weights = []
        lora1 = []
        for i in range(x.shape[0]):
            routing_weights.append(F.softmax(self.gate[i](x[i].view(-1, c)), dim=1, dtype=torch.float).view(bs_m//self.num_modals, n, -1))
            lora1.append(self.linear_b_fc1[i](self.linear_a_fc1[i](x[i])))
        lora1 = torch.stack(lora1)
        routing_weights = torch.stack(routing_weights)
        
        fused_lora1 = torch.zeros_like(x1)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if self.modality_mask[i][j] == 1:
                    per_routing_weights = routing_weights[i,j,:] * self.modality_mask[:,j].unsqueeze(0)
                    per_routing_weights /= (per_routing_weights.sum(dim=-1, keepdim=True) + 1e-6)
                    fused_lora1[i][j] = (lora1[:,j,:,:] * per_routing_weights.permute(1,0)[:,:,None]).sum(0)
        x1 = x1 + fused_lora1
        x1 = x1.view(bs_m, n, -1).contiguous()

        x1 = self.mlp[1](x1)
        x1 = x1.view(self.num_modals, bs_m//self.num_modals, n, -1).contiguous()
        x2 = self.mlp[2](x1).view(self.num_modals, bs_m//self.num_modals, n, -1).contiguous()

        lora2 = []
        for i in range(x.shape[0]):
            lora2.append(self.linear_b_fc2[i](self.linear_a_fc2[i](x1[i])))
        lora2 = torch.stack(lora2)
        
        fused_lora2 = torch.zeros_like(x2)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if self.modality_mask[i][j] == 1:
                    per_routing_weights = routing_weights[i,j,:] * self.modality_mask[:,j].unsqueeze(0)
                    per_routing_weights /= (per_routing_weights.sum(dim=-1, keepdim=True) + 1e-6)
                    fused_lora2[i][j] = (lora2[:,j,:,:] * per_routing_weights.permute(1,0)[:,:,None]).sum(0)
        x2 = x2 + fused_lora2
        x2 = x2.view(bs_m, n, -1).contiguous()
        
        return x2    
    
class MLPFusionHiRA(nn.Module):
    """Low-Rank Adaptation for the for Query (Q), Key (Q), Value (V) matrices"""

    def __init__(
        self,
        num_modals: int,
        mlp: nn.Module,
        i_linear_a_fc1: nn.Module,
        i_linear_b_fc1: nn.Module,
        i_linear_a_fc2: nn.Module,
        i_linear_b_fc2: nn.Module,
        d_linear_a_fc1: nn.Module,
        d_linear_b_fc1: nn.Module,
        d_linear_a_fc2: nn.Module,
        d_linear_b_fc2: nn.Module,
        e_linear_a_fc1: nn.Module,
        e_linear_b_fc1: nn.Module,
        e_linear_a_fc2: nn.Module,
        e_linear_b_fc2: nn.Module,
        l_linear_a_fc1: nn.Module,
        l_linear_b_fc1: nn.Module,
        l_linear_a_fc2: nn.Module,
        l_linear_b_fc2: nn.Module,
        gate: nn.ModuleList = None,
        modality_mask: torch.Tensor = None,
    ):
        super().__init__()
        self.num_modals = num_modals
        self.mlp = mlp
        self.linear_a_fc1 = nn.ModuleList([i_linear_a_fc1, d_linear_a_fc1, e_linear_a_fc1, l_linear_a_fc1])
        self.linear_b_fc1 = nn.ModuleList([i_linear_b_fc1, d_linear_b_fc1, e_linear_b_fc1, l_linear_b_fc1])
        self.linear_a_fc2 = nn.ModuleList([i_linear_a_fc2, d_linear_a_fc2, e_linear_a_fc2, l_linear_a_fc2])
        self.linear_b_fc2 = nn.ModuleList([i_linear_b_fc2, d_linear_b_fc2, e_linear_b_fc2, l_linear_b_fc2])
        self.gate = gate
        self.modality_mask = modality_mask

    def forward(self, x) -> torch.Tensor:
        bs_m, n, c = x.shape
        x1 = self.mlp.fc1(x).view(self.num_modals, bs_m//self.num_modals, n, -1).contiguous()
        
        x = x.view(self.num_modals, bs_m//self.num_modals, n, c).contiguous()
        routing_weights = []
        lora1 = []
        for i in range(x.shape[0]):
            routing_weights.append(F.softmax(self.gate[i](x[i].view(-1, c)), dim=1, dtype=torch.float).view(bs_m//self.num_modals, n, -1))
            # lora1.append(self.linear_b_fc1[i](self.linear_a_fc1[i](x[i])))
            lora_f1 = torch.matmul(
                self.linear_b_fc1[i].weight, self.linear_a_fc1[i].weight
            )  # [out, in]
            delta_f1_weight = self.mlp.fc1.weight * lora_f1  # Hadamard product [out, in]
            lora1.append(torch.matmul(x[i], delta_f1_weight.T))
            
        lora1 = torch.stack(lora1)
        routing_weights = torch.stack(routing_weights)
        
        fused_lora1 = torch.zeros_like(x1)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if self.modality_mask[i][j] == 1:
                    per_routing_weights = routing_weights[i,j,:] * self.modality_mask[:,j].unsqueeze(0)
                    per_routing_weights /= (per_routing_weights.sum(dim=-1, keepdim=True) + 1e-6)
                    fused_lora1[i][j] = (lora1[:,j,:,:] * per_routing_weights.permute(1,0)[:,:,None]).sum(0)
        x1 = x1 + fused_lora1
        x1 = x1.view(bs_m, n, -1).contiguous()

        x1 = self.mlp.act(x1)
        x1 = self.mlp.drop(x1)
        x1 = x1.view(self.num_modals, bs_m//self.num_modals, n, -1).contiguous()
        x2 = self.mlp.fc2(x1).view(self.num_modals, bs_m//self.num_modals, n, -1).contiguous()

        lora2 = []
        for i in range(x.shape[0]):
            # lora2.append(self.linear_b_fc2[i](self.linear_a_fc2[i](x1[i])))
            lora_f2 = torch.matmul(
                self.linear_b_fc2[i].weight, self.linear_a_fc2[i].weight
            )  # [out, in]
            delta_f2_weight = self.mlp.fc2.weight * lora_f2  # Hadamard product [out, in]
            lora2.append(torch.matmul(x1[i], delta_f2_weight.T))            
            
        lora2 = torch.stack(lora2)
        
        fused_lora2 = torch.zeros_like(x2)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if self.modality_mask[i][j] == 1:
                    per_routing_weights = routing_weights[i,j,:] * self.modality_mask[:,j].unsqueeze(0)
                    per_routing_weights /= (per_routing_weights.sum(dim=-1, keepdim=True) + 1e-6)
                    fused_lora2[i][j] = (lora2[:,j,:,:] * per_routing_weights.permute(1,0)[:,:,None]).sum(0)
        x2 = x2 + fused_lora2
        x2 = x2.view(bs_m, n, -1).contiguous()
        
        x2 = self.mlp.drop(x2)
        return x2    
    
    
class MLPFusionLoRA_wo_router(nn.Module):
    """Low-Rank Adaptation for the for Query (Q), Key (Q), Value (V) matrices"""

    def __init__(
        self,
        num_modals: int,
        mlp: nn.Module,
        i_linear_a_fc1: nn.Module,
        i_linear_b_fc1: nn.Module,
        i_linear_a_fc2: nn.Module,
        i_linear_b_fc2: nn.Module,
        d_linear_a_fc1: nn.Module,
        d_linear_b_fc1: nn.Module,
        d_linear_a_fc2: nn.Module,
        d_linear_b_fc2: nn.Module,
        e_linear_a_fc1: nn.Module,
        e_linear_b_fc1: nn.Module,
        e_linear_a_fc2: nn.Module,
        e_linear_b_fc2: nn.Module,
        l_linear_a_fc1: nn.Module,
        l_linear_b_fc1: nn.Module,
        l_linear_a_fc2: nn.Module,
        l_linear_b_fc2: nn.Module,
        modality_mask: torch.Tensor = None,
    ):
        super().__init__()
        self.num_modals = num_modals
        self.mlp = mlp
        self.linear_a_fc1 = nn.ModuleList([i_linear_a_fc1, d_linear_a_fc1, e_linear_a_fc1, l_linear_a_fc1])
        self.linear_b_fc1 = nn.ModuleList([i_linear_b_fc1, d_linear_b_fc1, e_linear_b_fc1, l_linear_b_fc1])
        self.linear_a_fc2 = nn.ModuleList([i_linear_a_fc2, d_linear_a_fc2, e_linear_a_fc2, l_linear_a_fc2])
        self.linear_b_fc2 = nn.ModuleList([i_linear_b_fc2, d_linear_b_fc2, e_linear_b_fc2, l_linear_b_fc2])
        self.modality_mask = modality_mask

    def forward(self, x) -> torch.Tensor:
        bs_m, n, c = x.shape
        x1 = self.mlp.fc1(x).view(self.num_modals, bs_m//self.num_modals, n, -1).contiguous()
        
        x = x.view(self.num_modals, bs_m//self.num_modals, n, c).contiguous()
        routing_weights = []
        lora1 = []
        for i in range(x.shape[0]):
            routing_weights.append(F.softmax(torch.ones([x[i].view(-1, c).shape[0], 4]), dim=1, dtype=torch.float).view(bs_m//self.num_modals, n, -1).to(x.device))
            lora1.append(self.linear_b_fc1[i](self.linear_a_fc1[i](x[i])))
        lora1 = torch.stack(lora1)
        routing_weights = torch.stack(routing_weights)
        
        fused_lora1 = torch.zeros_like(x1)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if self.modality_mask[i][j] == 1:
                    per_routing_weights = routing_weights[i,j,:] * self.modality_mask[:,j].unsqueeze(0)
                    per_routing_weights /= (per_routing_weights.sum(dim=-1, keepdim=True) + 1e-6)
                    fused_lora1[i][j] = (lora1[:,j,:,:] * per_routing_weights.permute(1,0)[:,:,None]).sum(0)
        x1 = x1 + fused_lora1
        x1 = x1.view(bs_m, n, -1).contiguous()

        x1 = self.mlp.act(x1)
        x1 = self.mlp.drop(x1)
        x1 = x1.view(self.num_modals, bs_m//self.num_modals, n, -1).contiguous()
        x2 = self.mlp.fc2(x1).view(self.num_modals, bs_m//self.num_modals, n, -1).contiguous()

        lora2 = []
        for i in range(x.shape[0]):
            lora2.append(self.linear_b_fc2[i](self.linear_a_fc2[i](x1[i])))
        lora2 = torch.stack(lora2)
        
        fused_lora2 = torch.zeros_like(x2)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if self.modality_mask[i][j] == 1:
                    per_routing_weights = routing_weights[i,j,:] * self.modality_mask[:,j].unsqueeze(0)
                    per_routing_weights /= (per_routing_weights.sum(dim=-1, keepdim=True) + 1e-6)
                    fused_lora2[i][j] = (lora2[:,j,:,:] * per_routing_weights.permute(1,0)[:,:,None]).sum(0)
        x2 = x2 + fused_lora2
        x2 = x2.view(bs_m, n, -1).contiguous()
        
        x2 = self.mlp.drop(x2)
        return x2    
    
class MultiModalLoRA_v2(nn.Module):
    """Low-Rank Adaptation for the for Query (Q), Key (Q), Value (V) matrices"""

    def __init__(
        self,
        num_modals: int,
        qkv: nn.Module,
        i_linear_a_q: nn.Module,
        i_linear_b_q: nn.Module,
        i_linear_a_v: nn.Module,
        i_linear_b_v: nn.Module,
        d_linear_a_q: nn.Module,
        d_linear_b_q: nn.Module,
        d_linear_a_v: nn.Module,
        d_linear_b_v: nn.Module,
        e_linear_a_q: nn.Module,
        e_linear_b_q: nn.Module,
        e_linear_a_v: nn.Module,
        e_linear_b_v: nn.Module,
        l_linear_a_q: nn.Module,
        l_linear_b_q: nn.Module,
        l_linear_a_v: nn.Module,
        l_linear_b_v: nn.Module,
        modality_mask: torch.Tensor = None,
    ):
        super().__init__()
        self.num_modals = num_modals
        self.qkv = qkv
        self.linear_a_q = nn.ModuleList([i_linear_a_q, d_linear_a_q, e_linear_a_q, l_linear_a_q])
        self.linear_b_q = nn.ModuleList([i_linear_b_q, d_linear_b_q, e_linear_b_q, l_linear_b_q])
        self.linear_a_v = nn.ModuleList([i_linear_a_v, d_linear_a_v, e_linear_a_v, l_linear_a_v])
        self.linear_b_v = nn.ModuleList([i_linear_b_v, d_linear_b_v, e_linear_b_v, l_linear_b_v])
        self.dim = qkv.in_features
        self.w_identity = torch.eye(self.dim)
        self.modality_mask = modality_mask   
    
    def forward(self, x) -> torch.Tensor:
        bs_m, n, c = x.shape
        x = x.view(self.num_modals, -1, n, c).contiguous()
        base_qkv = []
        shared_q = []
        shared_v = []
        specific_q = []
        specific_v = []
        for i in range(self.num_modals):
            # Compute the original qkv
            base_qkv.append(self.qkv(x[i]))  # Shape: (B, N, 3 * org_C)

            # Compute the new q and v components
            shared_q.append(self.linear_b_q[3](self.linear_a_q[3](x[i])))
            shared_v.append(self.linear_b_v[3](self.linear_a_v[3](x[i])))
            specific_q.append(self.linear_b_q[i](self.linear_a_q[i](x[i])))
            specific_v.append(self.linear_b_v[i](self.linear_a_v[i](x[i])))

        base_qkv = torch.stack(base_qkv)
        shared_q = torch.stack(shared_q)
        shared_v = torch.stack(shared_v)
        specific_q = torch.stack(specific_q)
        specific_v = torch.stack(specific_v)
        
        base_qkv[:, :, :, : self.dim] += shared_q
        base_qkv[:, :, :, -self.dim :] += shared_v
        
        base_qkv[:, :, :, : self.dim] += specific_q
        base_qkv[:, :, :, -self.dim :] += specific_v
        qkv = base_qkv.view(bs_m, n, -1).contiguous()
        return qkv
    

class MultiModalLoRA_v3(nn.Module):
    """Low-Rank Adaptation for the for Query (Q), Key (Q), Value (V) matrices"""

    def __init__(
        self,
        num_modals: int,
        qkv: nn.Module,
        i_linear_a_q: nn.Module,
        i_linear_o_q: nn.Module,
        i_linear_b_q: nn.Module,
        i_linear_a_v: nn.Module,
        i_linear_o_v: nn.Module,
        i_linear_b_v: nn.Module,
        d_linear_a_q: nn.Module,
        d_linear_o_q: nn.Module,
        d_linear_b_q: nn.Module,
        d_linear_a_v: nn.Module,
        d_linear_o_v: nn.Module,
        d_linear_b_v: nn.Module,
        e_linear_a_q: nn.Module,
        e_linear_o_q: nn.Module,
        e_linear_b_q: nn.Module,
        e_linear_a_v: nn.Module,
        e_linear_o_v: nn.Module,
        e_linear_b_v: nn.Module,
        l_linear_a_q: nn.Module,
        l_linear_o_q: nn.Module,
        l_linear_b_q: nn.Module,
        l_linear_a_v: nn.Module,
        l_linear_o_v: nn.Module,
        l_linear_b_v: nn.Module,
        modality_mask: torch.Tensor = None,
    ):
        super().__init__()
        self.num_modals = num_modals
        self.qkv = qkv
        self.linear_a_q = nn.ModuleList([i_linear_a_q, d_linear_a_q, e_linear_a_q, l_linear_a_q])
        self.linear_o_q = nn.ModuleList([i_linear_o_q, d_linear_o_q, e_linear_o_q, l_linear_o_q])
        self.linear_b_q = nn.ModuleList([i_linear_b_q, d_linear_b_q, e_linear_b_q, l_linear_b_q])
        self.linear_a_v = nn.ModuleList([i_linear_a_v, d_linear_a_v, e_linear_a_v, l_linear_a_v])
        self.linear_o_v = nn.ModuleList([i_linear_o_v, d_linear_o_v, e_linear_o_v, l_linear_o_v])
        self.linear_b_v = nn.ModuleList([i_linear_b_v, d_linear_b_v, e_linear_b_v, l_linear_b_v])
        self.dim = qkv.in_features
        self.w_identity = torch.eye(self.dim)
        self.modality_mask = modality_mask   
    
    def forward(self, x) -> torch.Tensor:
        bs_m, n, c = x.shape
        x = x.view(self.num_modals, -1, n, c).contiguous()
        base_qkv = []
        shared_q = []
        shared_v = []
        specific_q = []
        specific_v = []
        for i in range(self.num_modals):
            # Compute the original qkv
            base_qkv.append(self.qkv(x[i]))  # Shape: (B, N, 3 * org_C)

            # Compute the new q and v components
            shared_q.append(self.linear_b_q[3](F.gelu(self.linear_o_q[3](self.linear_a_q[3](x[i])))))
            shared_v.append(self.linear_b_v[3](F.gelu(self.linear_o_v[3](self.linear_a_v[3](x[i])))))
            specific_q.append(self.linear_b_q[i](F.gelu(self.linear_o_q[i](self.linear_a_q[i](x[i])))))
            specific_v.append(self.linear_b_v[i](F.gelu(self.linear_o_v[i](self.linear_a_v[i](x[i])))))

        base_qkv = torch.stack(base_qkv)
        shared_q = torch.stack(shared_q)
        shared_v = torch.stack(shared_v)
        specific_q = torch.stack(specific_q)
        specific_v = torch.stack(specific_v)
        
        base_qkv[:, :, :, : self.dim] += shared_q
        base_qkv[:, :, :, -self.dim :] += shared_v
        
        base_qkv[:, :, :, : self.dim] += specific_q
        base_qkv[:, :, :, -self.dim :] += specific_v
        qkv = base_qkv.view(bs_m, n, -1).contiguous()
        return qkv
    
    
class MultiModalLoRA_v4(nn.Module):
    """Low-Rank Adaptation for the for Query (Q), Key (Q), Value (V) matrices"""

    def __init__(
        self,
        num_modals: int,
        qkv: nn.Module,
        i_linear_a_q: nn.Module,
        i_linear_b_q: nn.Module,
        i_linear_a_v: nn.Module,
        i_linear_b_v: nn.Module,
        d_linear_a_q: nn.Module,
        d_linear_b_q: nn.Module,
        d_linear_a_v: nn.Module,
        d_linear_b_v: nn.Module,
        e_linear_a_q: nn.Module,
        e_linear_b_q: nn.Module,
        e_linear_a_v: nn.Module,
        e_linear_b_v: nn.Module,
        l_linear_a_q: nn.Module,
        l_linear_b_q: nn.Module,
        l_linear_a_v: nn.Module,
        l_linear_b_v: nn.Module,
    ):
        super().__init__()
        self.num_modals = num_modals
        self.qkv = qkv
        self.linear_a_q = nn.ModuleList([i_linear_a_q, d_linear_a_q, e_linear_a_q, l_linear_a_q])
        self.linear_b_q = nn.ModuleList([i_linear_b_q, d_linear_b_q, e_linear_b_q, l_linear_b_q])
        self.linear_a_v = nn.ModuleList([i_linear_a_v, d_linear_a_v, e_linear_a_v, l_linear_a_v])
        self.linear_b_v = nn.ModuleList([i_linear_b_v, d_linear_b_v, e_linear_b_v, l_linear_b_v])
        self.dim = qkv.in_features
        self.w_identity = torch.eye(self.dim)

    def forward(self, x) -> torch.Tensor:
        bs_m, n, c = x.shape
        x = x.view(self.num_modals, -1, n, c).contiguous()
        qkv = []
        shared_q = self.linear_b_q[3](self.linear_a_q[3](x.mean(0)))
        shared_v = self.linear_b_v[3](self.linear_a_v[3](x.mean(0)))
        for i in range(len(x)):
            # Compute the original qkv
            qkv.append(self.qkv(x[i]))  # Shape: (B, N, 3 * org_C)

            # Compute the new q and v components
            new_q = self.linear_b_q[i](self.linear_a_q[i](x[i])) + shared_q
            new_v = self.linear_b_v[i](self.linear_a_v[i](x[i])) + shared_v

            # Add new q and v components to the original qkv tensor
            qkv[i][:, :, : self.dim] += new_q
            qkv[i][:, :, -self.dim :] += new_v
        
        qkv = torch.stack(qkv).view(bs_m, n, -1).contiguous()
        return qkv    
    
    
class MultiModalLoRA_proj(nn.Module):
    """Low-Rank Adaptation for the for Query (Q), Key (Q), Value (V) matrices"""

    def __init__(
        self,
        num_modals: int,
        proj: nn.Module,
        i_linear_a_o: nn.Module,
        i_linear_b_o: nn.Module,
        d_linear_a_o: nn.Module,
        d_linear_b_o: nn.Module,
        e_linear_a_o: nn.Module,
        e_linear_b_o: nn.Module,
        l_linear_a_o: nn.Module,
        l_linear_b_o: nn.Module,
        modality_mask: torch.Tensor = None,
    ):
        super().__init__()
        self.num_modals = num_modals
        self.proj = proj
        self.linear_a_o = nn.ModuleList([i_linear_a_o, d_linear_a_o, e_linear_a_o, l_linear_a_o])
        self.linear_b_o = nn.ModuleList([i_linear_b_o, d_linear_b_o, e_linear_b_o, l_linear_b_o])
        self.dim = proj.in_features
        self.w_identity = torch.eye(self.dim)
        self.modality_mask = modality_mask   
    
    def forward(self, x) -> torch.Tensor:
        bs_m, n, c = x.shape
        x = x.view(self.num_modals, -1, n, c).contiguous()
        base_proj = []
        shared_proj = []
        specific_proj = []
        for i in range(self.num_modals):
            # Compute the original qkv
            base_proj.append(self.proj(x[i]))  # Shape: (B, N, 3 * org_C)

            # Compute the new q and v components
            shared_proj.append(self.linear_b_o[3](self.linear_a_o[3](x[i])))
            specific_proj.append(self.linear_b_o[i](self.linear_a_o[i](x[i])))

        base_proj = torch.stack(base_proj)
        shared_proj = torch.stack(shared_proj)
        specific_proj = torch.stack(specific_proj)
        
        base_proj += shared_proj
        base_proj += specific_proj
        
        proj = base_proj.view(bs_m, n, -1).contiguous()
        return proj
    
    
class MultiModalLoRA_kv(nn.Module):
    """Low-Rank Adaptation for the for Query (Q), Key (Q), Value (V) matrices"""

    def __init__(
        self,
        kv: nn.Module,
        i_linear_a_v: nn.Module,
        i_linear_b_v: nn.Module,
        d_linear_a_v: nn.Module,
        d_linear_b_v: nn.Module,
        e_linear_a_v: nn.Module,
        e_linear_b_v: nn.Module,
        l_linear_a_v: nn.Module,
        l_linear_b_v: nn.Module,
    ):
        super().__init__()
        self.kv = kv
        self.linear_a_v = nn.ModuleList([i_linear_a_v, d_linear_a_v, e_linear_a_v, l_linear_a_v])
        self.linear_b_v = nn.ModuleList([i_linear_b_v, d_linear_b_v, e_linear_b_v, l_linear_b_v])
        self.dim = kv.in_features
        self.w_identity = torch.eye(self.dim)

    def forward(self, x) -> torch.Tensor:
        bs_m, n, c = x.shape
        x = x.view(4, -1, n, c).contiguous()
        kv = []
        for i in range(len(x)):
            # Compute the original qkv
            kv.append(self.kv(x[i]))  # Shape: (B, N, 3 * org_C)

            # Compute the new q and v components
            new_v = self.linear_b_v[i](self.linear_a_v[i](x[i]))
            # new_v = self.linear_b_v[0](self.linear_a_v[0](x[i]))

            # Add new q and v components to the original qkv tensor
            kv[i][:, :, -self.dim :] += new_v

        kv = torch.stack(kv).view(bs_m, n, -1).contiguous()
        return kv
    

def svd_split_lora_init(W0: torch.Tensor, rank_lora: int):
    """
    对预训练权重 W0 做 SVD 分解，并用次主成分初始化 LoRA。
    
    Args:
        W0 (torch.Tensor): 预训练权重，shape [m, n]
        rank_lora (int): LoRA 使用的秩（从低奇异值中选）
    
    Returns:
        W_frozen (torch.Tensor): 主成分重建的 frozen 权重
        A (torch.Tensor): LoRA A 矩阵，shape [rank_lora, n]
        B (torch.Tensor): LoRA B 矩阵，shape [m, rank_lora]
    """
    # Step 1: SVD
    U, S, Vh = torch.linalg.svd(W0, full_matrices=False)  # W0 = U @ diag(S) @ Vh

    # Step 2: 主成分保留到 W_frozen
    r = rank_lora
    m, n = W0.shape

    # 主成分部分（前 m-r）
    U_main = U[:, :-r]
    S_main = S[:-r]
    Vh_main = Vh[:-r, :]
    W_frozen = (U_main * S_main) @ Vh_main  # 形状 [m, n]

    # 次主成分用于 LoRA 初始化（低秩）
    U_lora = U[:, -r:]                       # [m, r]
    S_lora = torch.sqrt(S[-r:])              # [r]
    Vh_lora = Vh[-r:, :]                     # [r, n]

    B = U_lora * S_lora  # [m, r]
    A = (S_lora.unsqueeze(-1) * Vh_lora).contiguous()  # [r, n]

    return W_frozen, A, B


def svd_residual_adaptation_three_part(W0: torch.Tensor, r: int):
    """
    输入:
        W0: [m, n] 原始预训练权重矩阵
        r: rank

    输出:
        W0_residual: [m, n] 冻结部分
        A1, B1: 中间主成分构建的可训练模块（输入先乘 A1，再乘 B1）
        A2, B2: 最后主成分构建的可训练模块（输入先乘 A2，再乘 B2）
    """
    assert W0.ndim == 2, "W0 must be a 2D weight matrix"
    U, S, Vh = torch.linalg.svd(W0, full_matrices=False)

    # 冻结 residual：前 m - 2r 个主成分
    U_res = U[:, :-2*r]
    S_res = S[:-2*r]
    Vh_res = Vh[:-2*r, :]
    W0_residual = (U_res * S_res) @ Vh_res  # shape: [m, n]

    # 中间 rank residual（用于 A1, B1）
    U1 = U[:, -2*r:-r]        # [m, r]
    S1 = S[-2*r:-r]           # [r]
    Vh1 = Vh[-2*r:-r, :]      # [r, n]
    sqrt_S1 = torch.sqrt(S1)
    A1 = nn.Parameter(sqrt_S1[:, None] * Vh1)   # [r, n]，输入先乘 A
    B1 = nn.Parameter(U1 * sqrt_S1)             # [m, r]，再乘 B

    # 最后 rank residual（用于 A2, B2）
    U2 = U[:, -r:]           # [m, r]
    S2 = S[-r:]              # [r]
    Vh2 = Vh[-r:, :]         # [r, n]
    sqrt_S2 = torch.sqrt(S2)
    A2 = nn.Parameter(sqrt_S2[:, None] * Vh2)   # [r, n]
    B2 = nn.Parameter(U2 * sqrt_S2)             # [m, r]

    return W0_residual, A1, B1, A2, B2


# Custom Implementation of the Mixtral Decoder with LoRA MoE
class NoisyTopkRouter(nn.Module):
    def __init__(self, hidden_dim, num_experts, bias=False):
        super().__init__()
        #layer for router logits
        self.topkroute_linear = nn.Linear(hidden_dim, num_experts, bias=bias)
        self.noise_linear =nn.Linear(hidden_dim, num_experts, bias=bias)

    def forward(self, hidden_states):
        # mh_ouput is the output tensor from multihead self attention block
        logits = self.topkroute_linear(hidden_states)

        #Noise logits
        noise_logits = self.noise_linear(hidden_states)

        #Adding scaled unit gaussian noise to the logits
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise

        return noisy_logits
    
    
class NoisyModalRouter_2D(nn.Module):
    def __init__(self, embed_dim, num_modals, bias=False):
        super().__init__()
        self.score_proj = nn.ModuleList([
            nn.Conv2d(embed_dim, 1, kernel_size=1, bias=bias) for _ in range(num_modals)
        ])
        self.noise_proj = nn.ModuleList([
            nn.Conv2d(embed_dim, 1, kernel_size=1, bias=bias) for _ in range(num_modals)
        ])

    def forward(self, feats):  # [M, B, C, H, W]
        scores = []
        for i in range(feats.shape[0]):
            score = self.score_proj[i](feats[i])  # [B, 1, H, W]
            noise = torch.randn_like(score) * F.softplus(self.noise_proj[i](feats[i]))
            scores.append(score + noise)
        scores = torch.stack(scores)  # [M, B, 1, H, W]
        weights = F.softmax(scores, dim=0)  # across modal dimension
        return weights