import torch

def kl_normal(mu, logvar):
    """KL( N(mu, diag(exp(logvar))) || N(0, I) ) -> [B]"""
    return 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=1)


def mse_loss(pred, target):
    """Mean squared error per-sample -> [B]"""
    return torch.mean((pred - target) ** 2, dim=1)