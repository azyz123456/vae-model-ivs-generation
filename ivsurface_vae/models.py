import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32, latent_dim: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.mu(h), self.logvar(h)


class PointwiseDecoder(nn.Module):
    """Decoder takes (z, T, delta) and outputs sigma(T, delta; z)."""
    def __init__(self, latent_dim: int, hidden: int = 32):
        super().__init__()
        self.mlp_inputs = nn.Sequential(
            nn.Linear(latent_dim + 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Softplus(),  # ensure positivity
        )


    def forward(self, z, T, delta):
        # z: [B, latent_dim], T, delta: [B, N] or [N]
        if T.ndim == 1:
            T = T.unsqueeze(0).repeat(z.size(0), 1)
            delta = delta.unsqueeze(0).repeat(z.size(0), 1)
        B, N = z.size(0), T.size(1)
        z_exp = z.unsqueeze(1).expand(B, N, z.size(1))
        inp = torch.cat([z_exp, T.unsqueeze(-1), delta.unsqueeze(-1)], dim=-1)  # [B, N, latent+2]
        out = self.mlp_inputs(inp).squeeze(-1)  # [B, N]
        return out


class PointwiseVAE(nn.Module):
    def __init__(self, maturities: torch.Tensor, deltas: torch.Tensor, latent_dim: int = 4, hidden: int = 32):
        super().__init__()
        self.maturities = maturities.clone().detach()
        self.deltas = deltas.clone().detach()
        self.M = len(maturities)
        self.D = len(deltas)
        in_dim = self.M * self.D
        self.enc = Encoder(in_dim=in_dim, hidden=hidden, latent_dim=latent_dim)
        self.dec = PointwiseDecoder(latent_dim=latent_dim, hidden=hidden)

        TT, DD = torch.meshgrid(self.maturities, self.deltas, indexing="ij")
        self.register_buffer("TT_grid", TT.reshape(-1))  # [M*D]
        self.register_buffer("DD_grid", DD.reshape(-1))  # [M*D]
        self.latent_dim = latent_dim

    def encode(self, x_flat):
        mu, logvar = self.enc(x_flat)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_grid(self, z):
        return self.dec(z, self.TT_grid.unsqueeze(0), self.DD_grid.unsqueeze(0))  # [B, M*D]

    def forward(self, x_flat):
        mu, logvar = self.encode(x_flat)
        z = self.reparameterize(mu, logvar)
        recon = self.decode_grid(z)
        return recon, mu, logvar