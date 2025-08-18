import torch
import numpy as np

class IVSurfaceDataset(torch.utils.data.Dataset):
    """
    surfaces: np.ndarray of shape [N, M, D] (annualized vols, e.g., 0.20 for 20%).
    """
    def __init__(self, surfaces: np.ndarray):
        assert surfaces.ndim == 3, "surfaces must be [N, M, D]"
        self.surfaces = torch.tensor(surfaces, dtype=torch.float32)

    def __len__(self):
        return self.surfaces.shape[0]

    def __getitem__(self, idx):
        s = self.surfaces[idx]           # [M, D]
        x_flat = s.reshape(-1)           # [M*D]
        return x_flat, s