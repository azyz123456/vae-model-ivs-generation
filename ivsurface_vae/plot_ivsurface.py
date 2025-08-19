"""
Plotting utilities for IV surface distributions saved by examples/train_and_condition.py

Usage:
  python -m examples.plot_ivsurface --in ivsurface_dist.npz --mode surface --out mean_surface.png
  python -m examples.plot_ivsurface --in ivsurface_dist.npz --mode slices --maturity_index 2 --out slices_t2.png

Modes:
  surface  : 2D heatmap of mean surface (T vs delta), with optional observed points overlay.
  slices   : line plot across delta at a chosen maturity index, with p05/p95 uncertainty band.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_npz(path):
    d = np.load(path)
    return {
        "mean": d["mean"],
        "p05": d["p05"],
        "p50": d["p50"],
        "p95": d["p95"],
        "mats": d["mats"],
        "deltas": d["deltas"],
        "T_obs": d.get("T_obs"),
        "D_obs": d.get("D_obs"),
        "S_obs": d.get("S_obs"),
    }


def plot_surface(mean, mats, deltas, T_obs=None, D_obs=None, out=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    # imshow expects [rows, cols] -> [mats, deltas]
    im = ax.imshow(
        mean,
        origin="lower",
        aspect="auto",
        extent=[deltas.min(), deltas.max(), mats.min(), mats.max()],
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Implied Volatility")
    ax.set_xlabel("Call Delta")
    ax.set_ylabel("Maturity (years)")
    ax.set_title("Mean Implied Vol Surface")

    # overlay observed points as black dots (if provided)
    if T_obs is not None and D_obs is not None:
        ax.scatter(D_obs, T_obs, s=30, edgecolor="k", facecolor="none")

    if out:
        plt.savefig(out, bbox_inches="tight", dpi=200)
    else:
        plt.show()



def plot_slices(mean, p05, p95, mats, deltas, maturity_index=0, T_obs=None, D_obs=None, S_obs=None, out=None):
    maturity_index = int(np.clip(maturity_index, 0, len(mats) - 1))
    t = mats[maturity_index]
    mu = mean[maturity_index]
    lo = p05[maturity_index]
    hi = p95[maturity_index]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(deltas, mu, label="mean")
    ax.fill_between(deltas, lo, hi, alpha=0.3, label="p05–p95")
    ax.set_xlabel("Call Delta")
    ax.set_ylabel("Implied Volatility")
    ax.set_title(f"Delta slice at T={t:.3f}y (index {maturity_index})")
    ax.legend()

    # overlay obs at matching maturity (within small tol)
    if T_obs is not None and D_obs is not None and S_obs is not None:
        mask = np.isclose(T_obs, t, rtol=0, atol=1e-6)
        if mask.any():
            ax.scatter(D_obs[mask], S_obs[mask], marker="x")

    if out:
        plt.savefig(out, bbox_inches="tight", dpi=200)
    else:
        plt.show()



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default="ivsurface_dist.npz")
    p.add_argument("--mode", choices=["surface", "slices"], default="surface")
    p.add_argument("--maturity_index", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    d = load_npz(args.inp)

    if args.mode == "surface":
        plot_surface(d["mean"], d["mats"], d["deltas"], T_obs=d.get("T_obs"), D_obs=d.get("D_obs"), out=args.out)
    else:
        plot_slices(d["mean"], d["p05"], d["p95"], d["mats"], d["deltas"], args.maturity_index, d.get("T_obs"), d.get("D_obs"), d.get("S_obs"), out=args.out)


if __name__ == "__main__":
    main()
