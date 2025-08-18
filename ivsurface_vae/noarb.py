import torch
import torch.nn.functional as F


def grid_to_surface(grid_vec: torch.Tensor, M: int, D: int) -> torch.Tensor:
    return grid_vec.view(-1, M, D)


def total_variance_surface(sig_surface: torch.Tensor, maturities: torch.Tensor) -> torch.Tensor:
    """w = t * sigma^2, broadcast across the grid."""
    B, M, D = sig_surface.shape
    t = maturities.view(1, M, 1).expand(B, M, D)
    return t * sig_surface ** 2


def calendar_arbitrage_penalty(sig_surface: torch.Tensor, maturities: torch.Tensor) -> torch.Tensor:
    """Penalty for negative \u2202w/\u2202t (monotonicity in maturity). Returns [B]."""
    w = total_variance_surface(sig_surface, maturities)
    dw_dt = w[:, 1:, :] - w[:, :-1, :]
    neg = F.relu(-dw_dt)
    return torch.mean(neg ** 2, dim=(1, 2))


def butterfly_arbitrage_penalty(sig_surface: torch.Tensor) -> torch.Tensor:
    """Convexity proxy along moneyness using second differences. Returns [B]."""
    w = sig_surface ** 2
    w_xx = w[:, :, 2:] - 2 * w[:, :, 1:-1] + w[:, :, :-2]
    neg = F.relu(-w_xx)
    return torch.mean(neg ** 2, dim=(1, 2))