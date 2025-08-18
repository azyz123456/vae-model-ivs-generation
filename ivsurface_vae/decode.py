from typing import Dict
import torch


def decode_distribution(model, z_samples: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Given z_samples [S, latent_dim], return a dict with
    mean / p05 / p50 / p95 surfaces of shape [M, D].
    """
    device = next(model.parameters()).device
    with torch.no_grad():
        z_samples = z_samples.to(device)
        grid_pred = model.decode_grid(z_samples)  # [S, M*D]
        S, _ = grid_pred.shape
        M, D = model.M, model.D
        surf_stack = grid_pred.view(S, M, D).cpu()
        mean = surf_stack.mean(dim=0)
        p05 = torch.quantile(surf_stack, 0.05, dim=0)
        p50 = torch.quantile(surf_stack, 0.50, dim=0)
        p95 = torch.quantile(surf_stack, 0.95, dim=0)
    return {"mean": mean, "p05": p05, "p50": p50, "p95": p95, "samples": surf_stack}