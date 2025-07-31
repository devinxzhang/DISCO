import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialGatedFusion(nn.Module):
    def __init__(self, in_channels, hidden_dim=128, num_modalities=3):
        super().__init__()
        self.num_modalities = num_modalities

        # 为每个模态定义一个 gating 网络
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, 1, kernel_size=1),
                nn.Sigmoid()
            )
            for _ in range(num_modalities)
        ])

    def forward(self, feats, modality_mask=None):
        """
        feats: Tensor of shape [M, B, C, H, W]
        modality_mask: Optional Tensor of shape [B, M] (binary 0/1)
        """
        M, B, C, H, W = feats.shape
        assert M == self.num_modalities

        gated_feats = []

        for i in range(M):
            f = feats[i]                         # [B, C, H, W]
            gate = self.gates[i](f)              # [B, 1, H, W]
            print(i, '--', gate.mean().item())
            f_weighted = f * gate                # gated feature

            if modality_mask is not None:
                # [B, 1, 1, 1] → broadcast 到 spatial
                mask = modality_mask[:, i].view(B, 1, 1, 1)
                f_weighted = f_weighted * mask

            gated_feats.append(f_weighted)

        # Stack & Sum → [M, B, C, H, W] → [B, C, H, W]
        fused = torch.stack(gated_feats, dim=0).sum(dim=0)
        return fused  # shape: [B, C, H, W]


class OTAligner(nn.Module):
    def __init__(self, cost_metric='l2', sinkhorn_iter=50, epsilon=0.05, return_transport=False):
        """
        Optimal Transport-based token alignment module.

        Args:
            cost_metric (str): 'l2' or 'cosine' distance for cost matrix
            sinkhorn_iter (int): number of Sinkhorn iterations
            epsilon (float): entropy regularization coefficient
            return_transport (bool): whether to return transport plan T
        """
        super().__init__()
        self.cost_metric = cost_metric
        self.sinkhorn_iter = sinkhorn_iter
        self.epsilon = epsilon
        self.return_transport = return_transport

    def compute_cost(self, A, B):
        """
        Compute pairwise cost between A and B.

        A: [B, N, C]
        B: [B, M, C]
        Returns: cost matrix [B, N, M]
        """
        if self.cost_metric == 'l2':
            cost = (A.unsqueeze(2) - B.unsqueeze(1)).pow(2).sum(-1)
        elif self.cost_metric == 'cosine':
            A_norm = F.normalize(A, dim=-1)
            B_norm = F.normalize(B, dim=-1)
            cost = 1 - torch.bmm(A_norm, B_norm.transpose(1, 2))  # [B, N, M]
        else:
            raise ValueError("Unknown cost metric: choose 'l2' or 'cosine'")
        return cost

    def sinkhorn(self, cost, epsilon, n_iter):
        """
        Sinkhorn algorithm to compute soft transport plan.

        Args:
            cost: [B, N, M]
        Returns:
            T: [B, N, M] transport matrix
        """
        B, N, M = cost.shape
        mu = torch.full((B, N), 1.0 / N, device=cost.device)
        nu = torch.full((B, M), 1.0 / M, device=cost.device)

        u = torch.ones_like(mu)
        v = torch.ones_like(nu)
        K = torch.exp(-cost / epsilon)  # [B, N, M]

        for _ in range(n_iter):
            Kv = torch.bmm(K, v.unsqueeze(-1)).squeeze(-1)
            Kv = torch.nan_to_num(Kv, nan=1.0, posinf=1.0, neginf=1.0)
            Kv = Kv.clamp(min=1e-8)
            u = mu / Kv

            Ku = torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1)
            Ku = torch.nan_to_num(Ku, nan=1.0, posinf=1.0, neginf=1.0)
            Ku = Ku.clamp(min=1e-8)
            v = nu / Ku

        T = u.unsqueeze(-1) * K * v.unsqueeze(1)  # [B, N, M]
        return T

    def forward(self, feat_a, feat_b):
        """
        Forward pass for OT-based feature alignment.

        Args:
            feat_a: [B, N, C] — anchor modality
            feat_b: [B, M, C] — to be aligned to feat_a
        Returns:
            aligned_feat_b: [B, N, C]
            (optional) transport matrix T: [B, N, M]
        """
        cost = self.compute_cost(feat_a, feat_b)       # [B, N, M]
        T = self.sinkhorn(cost, self.epsilon, self.sinkhorn_iter)
        T = torch.clamp(T, min=1e-8)
        aligned = torch.bmm(T, feat_b)                 # [B, N, C]

        if self.return_transport:
            return aligned, T
        else:
            return aligned
        

def modality_agnostic_fusion(feature):
    """
    实现论文图中 Fig.3 所描述的 Modality-Agnostic Feature Fusion 操作（无参数版）
    
    Args:
        feature: Tensor [m, b, n, c]，多模态特征（m 是 modality 个数）
        feature_pre: Tensor [b, n, c]，上一步的 f_ma^{t-1}，即 modality-agnostic feature

    Returns:
        f_ma: Tensor [b, n, c]，融合后的新 modality-agnostic feature
    """
    m, b, n, c = feature.shape
    feature_pre = feature.mean(dim=0)

    # Step 1: cosine similarity between each modality and f_ma^{t-1}
    feat_norm = F.normalize(feature, dim=-1)          # [m, b, n, c]
    pre_norm  = F.normalize(feature_pre, dim=-1)       # [b, n, c]
    sim = (feat_norm * pre_norm.unsqueeze(0)).sum(-1, keepdim=True)  # [m, b, n, 1]

    # Step 2: reweight each modality feature
    weighted_feat = feature * sim                     # [m, b, n, c]

    # Step 3: average fusion to get f_mb
    f_mb = weighted_feat.mean(dim=0)                  # [b, n, c]

    # Step 4: similarity between each modality and f_mb
    f_mb_norm = F.normalize(f_mb, dim=-1)             # [b, n, c]
    cs = (F.normalize(feature, dim=-1) * f_mb_norm.unsqueeze(0)).sum(-1)  # [m, b, n]

    # Step 5: element-wise max selection (hard routing)
    idx = cs.argmax(dim=0)                            # [b, n]，每个token选择最相似模态索引

    # Step 6: gather modality-specific selected feature
    feature_perm = feature.permute(1, 2, 0, 3)  # [b, n, m, c]
    idx_expand = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, c)  # [b, n, 1, c]
    f_ms = torch.gather(feature_perm, dim=2, index=idx_expand).squeeze(2)  # [b, n, c]

    # Step 7: residual connection
    f_ma = f_mb + f_ms                         # [b, n, c]

    return f_ma
