from typing import Dict, List
import torch
from torch.optim import Adam
from .losses import kl_normal, mse_loss
from .noarb import grid_to_surface, calendar_arbitrage_penalty, butterfly_arbitrage_penalty


def train_vae(model, dataset, cfg) -> Dict[str, List[float]]:
    """Train the PointwiseVAE with optional no-arbitrage penalties."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    model.to(cfg.device)
    opt = Adam(model.parameters(), lr=cfg.lr)
    hist = {"loss": [], "recon": [], "kl": [], "cal": [], "bfly": []}

    maturities = model.maturities.to(cfg.device)

    for epoch in range(cfg.epochs):
        running = {k: 0.0 for k in hist}
        for x_flat, x_surf in loader:
            x_flat = x_flat.to(cfg.device)
            x_surf = x_surf.to(cfg.device)

            opt.zero_grad()
            recon_flat, mu, logvar = model(x_flat)
            recon_surf = grid_to_surface(recon_flat, model.M, model.D)

            recon_err = mse_loss(recon_flat, x_flat)
            kl = kl_normal(mu, logvar)
            loss = recon_err + cfg.beta * kl

            cal_pen = recon_err.new_zeros(recon_err.shape)
            bfly_pen = recon_err.new_zeros(recon_err.shape)
            if cfg.lambda_cal > 0.0:
                cal_pen = calendar_arbitrage_penalty(recon_surf, maturities)
                loss = loss + cfg.lambda_cal * cal_pen
            if cfg.lambda_bfly > 0.0 and model.D >= 3:
                bfly_pen = butterfly_arbitrage_penalty(recon_surf)
                loss = loss + cfg.lambda_bfly * bfly_pen

            loss = loss.mean()
            loss.backward()
            opt.step()

            B = x_flat.size(0)
            running["loss"] += loss.item() * B
            running["recon"] += recon_err.mean().item() * B
            running["kl"] += kl.mean().item() * B
            running["cal"] += cal_pen.mean().item() * B
            running["bfly"] += bfly_pen.mean().item() * B

        N = len(loader.dataset) - (len(loader.dataset) % cfg.batch_size)
        for k in hist:
            hist[k].append(running[k] / max(N, 1))
        if cfg.verbose:
            print(
                f"Epoch {epoch+1:03d} | loss {hist['loss'][-1]:.6f} | recon {hist['recon'][-1]:.6f} | "
                f"kl {hist['kl'][-1]:.6f} | cal {hist['cal'][-1]:.6f} | bfly {hist['bfly'][-1]:.6f}"
            )
    return hist