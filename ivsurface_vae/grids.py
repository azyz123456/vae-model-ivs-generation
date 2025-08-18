import torch

def default_maturity_grid():
    """Example maturities (years): 1w, 1m, 2m, 3m, 6m, 9m, 1y, 3y."""
    return torch.tensor([1/52, 1/12, 2/12, 3/12, 6/12, 9/12, 1.0, 3.0], dtype=torch.float32)


def default_delta_grid():
    """Example call-delta grid."""
    return torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90], dtype=torch.float32)