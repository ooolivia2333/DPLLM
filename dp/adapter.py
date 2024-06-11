import torch

def make_adapter(in_dim, bottleneck_dim, out_dim):
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, bottleneck_dim),
        torch.nn.GELU(),
        torch.nn.Linear(bottleneck_dim, out_dim),
    )

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)