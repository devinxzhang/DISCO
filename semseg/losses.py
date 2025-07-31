import torch
from torch import nn, Tensor
from torch.nn import functional as F

def compute_batch_class_weights(target, num_classes, ignore_index=255, epsilon=1e-6, max_weight=10.0, default_weight=1.0):
    """
    target: [B, H, W] tensor with class labels
    return: weight tensor of shape [num_classes], where:
        - Appeared classes get inverse-frequency weight (clipped)
        - Absent classes get default weight (usually 1.0)
    """
    mask = (target != ignore_index)
    valid_target = target[mask]

    class_counts = torch.bincount(valid_target, minlength=num_classes).float()
    appeared_mask = class_counts > 0

    weights = torch.full((num_classes,), default_weight, dtype=torch.float, device=target.device)
    appeared_counts = class_counts[appeared_mask]
    inverse_weights = 1.0 / (appeared_counts + epsilon)
    inverse_weights = inverse_weights / inverse_weights.min()
    inverse_weights = torch.clamp(inverse_weights, max=max_weight)
    weights[appeared_mask] = inverse_weights
    
    return weights


class CrossEntropy(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None, aux_weights: list = [1, 0.4, 0.4]) -> None:
        super().__init__()
        self.aux_weights = aux_weights
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        return self.criterion(preds, labels)

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None, thresh: float = 0.7, aux_weights: list = [0.25, 0.25, 0.25, 0.25, 1]) -> None:
        super().__init__()
        self.ignore_label = ignore_label
        self.aux_weights = aux_weights
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction='none')

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        ################
        weights = compute_batch_class_weights(labels, preds.shape[1], self.ignore_label)
        self.criterion.weight = weights
        ################
        n_min = labels[labels != self.ignore_label].numel() // 16
        loss = self.criterion(preds, labels).view(-1)
        loss_hard = loss[loss > self.thresh]

        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)

        return torch.mean(loss_hard)

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)


class Dice(nn.Module):
    def __init__(self, ignore_label: int = 255, delta: float = 0.5, aux_weights: list = [1, 0.4, 0.4]):
        """
        delta: Controls weight given to FP and FN. This equals to dice score when delta=0.5
        """
        super().__init__()
        self.ignore_label = ignore_label
        self.delta = delta
        self.aux_weights = aux_weights
        self.weight = torch.tensor([2, 1, 1, 2, 2, 2, 1, 3, 2, 10, 15, 1, 1, 1, 1, 4, 1, 1, 10, 1], dtype=torch.float).cuda()

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        num_classes = preds.shape[1]

        # 标记 mask 区域
        ignore_mask = labels == self.ignore_label
        labels = labels.clone()
        labels[ignore_mask] = 0  # 临时设置成合法类别索引
        
        # One-hot 编码（现在 labels 中最大不会超过 num_classes-1）
        one_hot = F.one_hot(labels, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # 构造 broadcast 掩码并应用到 both preds 和 one_hot
        mask = (~ignore_mask).unsqueeze(1)  # [B, 1, H, W]
        labels = one_hot * mask
        preds = preds * mask

        tp = torch.sum(labels*preds, dim=(2, 3))
        fn = torch.sum(labels*(1-preds), dim=(2, 3))
        fp = torch.sum((1-labels)*preds, dim=(2, 3))

        dice_score = (tp + 1e-6) / (tp + self.delta * fn + (1 - self.delta) * fp + 1e-6)
        dice_score = dice_score * self.weight.view(1, num_classes)
        dice_score = torch.sum(1 - dice_score, dim=-1)

        dice_score = dice_score / num_classes
        return dice_score.mean()

    def forward(self, preds, targets: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, targets) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, targets)


__all__ = ['CrossEntropy', 'OhemCrossEntropy', 'Dice']


def get_loss(loss_fn_name: str = 'CrossEntropy', ignore_label: int = 255, cls_weights: Tensor = None):
    assert loss_fn_name in __all__, f"Unavailable loss function name >> {loss_fn_name}.\nAvailable loss functions: {__all__}"
    if loss_fn_name == 'Dice':
        return Dice(ignore_label)
    return eval(loss_fn_name)(ignore_label, cls_weights)


if __name__ == '__main__':
    pred = torch.randint(0, 19, (2, 19, 480, 640), dtype=torch.float)
    label = torch.randint(0, 19, (2, 480, 640), dtype=torch.long)
    loss_fn = Dice()
    y = loss_fn(pred, label)
    print(y)