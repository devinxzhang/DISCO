from torch import nn
from torch.optim import AdamW, SGD


def get_optimizer(model: nn.Module, optimizer: str, lr: float, weight_decay: float = 0.01):
    wd_params, nwd_params = [], []
    for name, p in model.named_parameters():
        if 'backbone' in name and \
            'fpn' not in name and \
            'linear_a_q' not in name and 'linear_b_q' not in name and 'linear_o_q' not in name and \
            'linear_a_k' not in name and 'linear_b_k' not in name and 'linear_o_k' not in name and \
            'linear_a_v' not in name and 'linear_b_v' not in name and 'linear_o_v' not in name and \
            'linear_a_fc1' not in name and 'linear_b_fc1' not in name and 'linear_a_fc2' not in name and \
            'linear_b_fc2' not in name and 'gate' not in name:
            p.requires_grad = False
        if p.requires_grad:
            if p.dim() == 1:
                nwd_params.append(p)
            else:
                wd_params.append(p)
    
    params = [
        {"params": wd_params},
        {"params": nwd_params, "weight_decay": 0}
    ]

    if optimizer == 'adamw':
        return AdamW(params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    else:
        return SGD(params, lr, momentum=0.9, weight_decay=weight_decay)