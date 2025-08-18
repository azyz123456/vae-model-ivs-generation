from typing import Tuple
import math
import numpy as np
import torch
from torch.optim import LBFGS


@torch.no_grad()
def _posterior_energy(model, z, T_obs_t, D_obs_t, S_obs_t, tau: float) -> torch.Tensor:
    pred = model.dec(z, T_obs_t, D_obs_t).squeeze(0)  # [N_obs]
    resid = (pred - S_obs_t)
    data = (resid ** 2).sum() / (2 * tau ** 2)
    prior = 0.5 * torch.sum(z ** 2)
    return data + prior


def _posterior_energy_and_grad(model, z, T_obs_t, D_obs_t, S_obs_t, tau: float):
    z = z.requires_grad_(True)
    pred = model.dec(z, T_obs_t, D_obs_t).squeeze(0)
    resid = (pred - S_obs_t)
    data = (resid ** 2).sum() / (2 * tau ** 2)
    prior = 0.5 * torch.sum(z ** 2)
    U = data + prior
    grad = torch.autograd.grad(U, z)[0]
    return U.detach(), grad.detach()


def sample_latent_sgld(model, T_obs: np.ndarray, Delta_obs: np.ndarray, sigma_obs: np.ndarray, cfg) -> torch.Tensor:
    """SGLD samples of z | obs. Returns [n_samples, latent_dim]."""
    torch.manual_seed(cfg.seed)
    model = model.to(cfg.device).eval()

    T_obs_t = torch.tensor(T_obs, dtype=torch.float32, device=cfg.device).unsqueeze(0)
    D_obs_t = torch.tensor(Delta_obs, dtype=torch.float32, device=cfg.device).unsqueeze(0)
    S_obs_t = torch.tensor(sigma_obs, dtype=torch.float32, device=cfg.device)

    z = torch.randn(1, model.latent_dim, device=cfg.device)
    samples = []
    total_iters = cfg.burn_in + cfg.n_samples * cfg.thin

    for it in range(total_iters):
        _, grad = _posterior_energy_and_grad(model, z, T_obs_t, D_obs_t, S_obs_t, cfg.obs_noise)
        eps = cfg.step_size
        noise = torch.randn_like(z)
        z = z - 0.5 * eps * grad + math.sqrt(eps) * noise
        if it >= cfg.burn_in and ((it - cfg.burn_in) % cfg.thin == 0):
            samples.append(z.detach().cpu().clone())

    return torch.vstack(samples) if samples else torch.empty(0, model.latent_dim)


def calibrate_z_map_lbfgs(model, T_obs: np.ndarray, Delta_obs: np.ndarray, sigma_obs: np.ndarray, tau: float = 0.02, max_iter: int = 200) -> torch.Tensor:
    device = next(model.parameters()).device
    T_obs_t = torch.tensor(T_obs, dtype=torch.float32, device=device).unsqueeze(0)
    D_obs_t = torch.tensor(Delta_obs, dtype=torch.float32, device=device).unsqueeze(0)
    S_obs_t = torch.tensor(sigma_obs, dtype=torch.float32, device=device)

    z = torch.zeros(1, model.latent_dim, device=device, requires_grad=True)
    opt = LBFGS([z], max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        pred = model.dec(z, T_obs_t, D_obs_t).squeeze(0)
        data = ((pred - S_obs_t) ** 2).sum() / (2 * tau ** 2)
        prior = 0.5 * torch.sum(z ** 2)
        loss = data + prior
        loss.backward()
        return loss

    opt.step(closure)
    return z.detach().cpu()