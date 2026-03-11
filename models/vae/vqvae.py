import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# VQ MODULE
# ============================================================
class VectorQuantizer(nn.Module):
    """
    Standard VQ-VAE quantizer with straight-through estimator.

    z_e: (B, D, H, W)
    returns:
      z_q: (B, D, H, W)
      indices: (B, H, W)
      vq_loss: scalar
    """
    def __init__(self, num_codes: int, code_dim: int, beta: float = 0.25):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.beta = beta

        self.codebook = nn.Embedding(num_codes, code_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)

    def forward(self, z_e):
        B, D, H, W = z_e.shape

        # (B, H, W, D) -> (BHW, D)
        z = z_e.permute(0, 2, 3, 1).contiguous()
        flat = z.view(-1, D)

        # compute distances to codebook: ||x - e||^2
        # dist = x^2 - 2 x e + e^2
        cb = self.codebook.weight  # (K, D)
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ cb.t()
            + cb.pow(2).sum(1).unsqueeze(0)
        )  # (BHW, K)

        indices = torch.argmin(dist, dim=1)  # (BHW,)
        z_q = self.codebook(indices).view(B, H, W, D)  # (B, H, W, D)

        # losses
        codebook_loss = F.mse_loss(z_q, z.detach())
        commit_loss = F.mse_loss(z_q.detach(), z)
        vq_loss = codebook_loss + self.beta * commit_loss

        # straight-through estimator
        z_q_st = z + (z_q - z).detach()

        # back to (B, D, H, W)
        z_q_st = z_q_st.permute(0, 3, 1, 2).contiguous()
        indices = indices.view(B, H, W)

        return z_q_st, indices, vq_loss


# ============================================================
# VQ-VAE (LEVEL)
# ============================================================
class Encoder(nn.Module):
    """
    16x16 -> 4x4 token map (downsample twice)
    """
    def __init__(self, n_channel: int, code_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_channel, 64, 4, stride=2, padding=1),  # 16 -> 8
            nn.ReLU(),
            nn.Conv2d(64, code_dim, 4, stride=2, padding=1),      # 8 -> 4
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)  # (B, D, 4, 4)


class Decoder(nn.Module):
    """
    4x4 -> 16x16 (upsample twice)
    Condition injection: CLIP embedding -> code_dim and broadcast add
    Output: logits (no sigmoid)
    """
    def __init__(self, n_channel: int, code_dim: int, clip_dim: int):
        super().__init__()
        self.fc_clip = nn.Linear(clip_dim, code_dim)

        self.net = nn.Sequential(
            nn.ConvTranspose2d(code_dim, 64, 4, stride=2, padding=1),  # 4 -> 8
            nn.ReLU(),
            nn.ConvTranspose2d(64, n_channel, 4, stride=2, padding=1),  # 8 -> 16
            # no sigmoid: we'll use BCEWithLogits
        )

    def forward(self, z_q, c):
        # c: (B, clip_dim) -> (B, code_dim, 1, 1)
        c_proj = self.fc_clip(c).unsqueeze(-1).unsqueeze(-1)
        z = z_q + c_proj
        return self.net(z)  # logits (B,1,16,16)





class NaiveVQVAE(nn.Module):
    def __init__(self, n_channel: int, num_codes: int, code_dim: int, beta_vq: float, cond_dim: int = 64):
        super().__init__()
        self.enc = Encoder(n_channel=n_channel, code_dim=code_dim)
        self.vq = VectorQuantizer(num_codes, code_dim, beta=beta_vq)
        self.dec = Decoder(n_channel=n_channel, code_dim=code_dim, clip_dim=cond_dim)

        self.cond_embed = nn.Sequential(
            nn.Linear(512, cond_dim),
            nn.LayerNorm(cond_dim),
            nn.ReLU(),
        )


    def forward(self, x, c):
        z_e = self.enc(x)
        z_q, indices, vq_loss = self.vq(z_e)

        c_emb = self.cond_embed(c)

        logits = self.dec(z_q, c_emb)
        return logits, indices, vq_loss

