import torch

def normalize_lastdim(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if v.dim() == 1:
        return v / (v.norm(p=2) + eps)
    return v / (v.norm(p=2, dim=-1, keepdim=True) + eps)