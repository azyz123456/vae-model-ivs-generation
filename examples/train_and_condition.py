"""Minimal end-to-end example using synthetic data."""
import numpy as np
import torch

from ivsurface_vae import (
    TrainCfg,
    PosteriorCfg,
    default_maturity_grid,
    default_delta_grid,
    IVSurfaceDataset,
    PointwiseVAE,
    train_vae,
    sample_latent_sgld,
    calibrate_z_map_lbfgs,
    decode_distribution,
)


def make_synthetic_dataset(N=2000, seed=123):
    rng = np.random.default_rng(seed)
    MATS = default_maturity_grid()
    DELTAS = default_delta_grid()
    M, D = len(MATS), len(DELTAS)

    base = rng.uniform(0.10, 0.30, size=(N, 1, 1))
    term = np.linspace(1.2, 0.8, M).reshape(1, M, 1)
    skew = np.linspace(0.85, 1.15, D).reshape(1, 1, D)
    noise = rng.normal(0, 0.01, size=(N, M, D))
    surf = np.clip(base * term * skew + noise, 0.05, 1.0)
    return MATS, DELTAS, IVSurfaceDataset(surf)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) data
    MATS, DELTAS, dataset = make_synthetic_dataset(N=2000)

    # 2) model + train
    vae = PointwiseVAE(MATS, DELTAS, latent_dim=4, hidden=32)
    cfg = TrainCfg(beta=1.0, lambda_cal=0.0, lambda_bfly=0.0, epochs=20, lr=1e-3, device=device)
    train_vae(vae, dataset, cfg)

    # 3) pick one surface and pretend we only observe a subset
    x_flat, true_surface = dataset[10]
    mat_mask = (MATS <= 1.0)
    del_mask = (DELTAS >= 0.25) & (DELTAS <= 0.75)
    mat_idx = torch.where(mat_mask)[0].numpy()
    del_idx = torch.where(del_mask)[0].numpy()

    grid_idx = [(int(m), int(d)) for m in mat_idx for d in del_idx]
    rng = np.random.default_rng(7)
    rng.shuffle(grid_idx)
    obs_pairs = grid_idx[:10]

    T_obs = np.array([MATS[m].item() for (m, d) in obs_pairs], dtype=np.float32)
    D_obs = np.array([DELTAS[d].item() for (m, d) in obs_pairs], dtype=np.float32)
    S_obs = np.array([true_surface[m, d].item() for (m, d) in obs_pairs], dtype=np.float32)

    # 4a) MAP estimate of latent z
    z_map = calibrate_z_map_lbfgs(vae, T_obs, D_obs, S_obs, tau=0.02, max_iter=150)
    print("MAP z:", z_map.shape)

    # 4b) Posterior samples via SGLD
    post_cfg = PosteriorCfg(obs_noise=0.02, n_samples=200, burn_in=200, thin=2, step_size=1e-3, device=device)
    z_samples = sample_latent_sgld(vae, T_obs, D_obs, S_obs, post_cfg)
    print("Posterior z samples:", z_samples.shape)

    # 5) Decode to distribution over full surfaces
    dist = decode_distribution(vae, z_samples)
    print({k: v.shape for k, v in dist.items() if k != "samples"})


if __name__ == "__main__":
    main()