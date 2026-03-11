import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

from models.vae.base import ResBlock


class ResMapEncoder(nn.Module):
    """
    16x16 → 4x4 with rich structure
    """
    def __init__(self, n_channel, code_dim, drop_rate=0.1):
        super().__init__()

        self.stem = nn.Conv2d(n_channel, 64, 3, padding=1)

        self.res1 = ResBlock(64, 128, drop_rate, use_se=True)
        self.down1 = nn.Conv2d(128, 128, 4, stride=2, padding=1)  # 16 → 8

        self.res2 = ResBlock(128, 256, drop_rate, use_se=True)
        self.down2 = nn.Conv2d(256, code_dim, 4, stride=2, padding=1)  # 8 → 4

    def forward(self, x):
        x = self.stem(x)
        x = self.res1(x)
        x = self.down1(x)
        x = self.res2(x)
        x = self.down2(x)
        return x  # (B, D, 4, 4)

class EMAVectorQuantizer(nn.Module):
    def __init__(self, num_codes, code_dim, beta=0.25, decay=0.99, eps=1e-5):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.beta = beta
        self.decay = decay
        self.eps = eps

        self.codebook = nn.Embedding(num_codes, code_dim)
        self.codebook.weight.data.uniform_(-1 / num_codes, 1 / num_codes)

        self.register_buffer("cluster_size", torch.zeros(num_codes))
        self.register_buffer("embed_avg", self.codebook.weight.data.clone())

    def forward(self, z_e):
        B, D, H, W = z_e.shape
        assert D == self.code_dim, f"z_e dim {D} != code_dim {self.code_dim}"

        # (B, D, H, W) → (BHW, D)
        z = z_e.permute(0, 2, 3, 1).contiguous().view(-1, D)

        # distances
        dist = (
            z.pow(2).sum(1, keepdim=True)
            - 2 * z @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(1)
        )
        indices = dist.argmin(1)              # (BHW,)
        z_q = self.codebook(indices)           # (BHW, D)

        # ===== EMA update =====
        if self.training:
            onehot = F.one_hot(indices, self.num_codes).type_as(z)

            # Detach from gradient graph via detach()
            self.cluster_size.mul_(self.decay).add_(
                onehot.sum(0).detach(), alpha=1 - self.decay
            )

            embed_sum = (z.t() @ onehot).t()   # (num_codes, D)
            self.embed_avg.mul_(self.decay).add_(
                embed_sum.detach(), alpha=1 - self.decay
            )

            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps)
                / (n + self.num_codes * self.eps)
                * n
            )

            self.codebook.weight.data.copy_(
                self.embed_avg / cluster_size.unsqueeze(1)
            )

        # ===== VQ loss =====
        commit_loss = F.mse_loss(z_q.detach(), z)
        vq_loss = self.beta * commit_loss

        # ===== straight-through =====
        z_q_st = z + (z_q - z).detach()        # (BHW, D)

        # back to (B, D, H, W)
        z_q_st = z_q_st.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()
        indices = indices.view(B, H, W)

        return z_q_st, indices, vq_loss


class FiLM(nn.Module):
    def __init__(self, cond_dim, feat_dim):
        super().__init__()
        self.fc = nn.Linear(cond_dim, feat_dim * 2)

    def forward(self, x, c):
        gamma, beta = self.fc(c).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta

class ResDecoder(nn.Module):
    def __init__(self, n_channel, code_dim, cond_dim):
        super().__init__()
        self.film = FiLM(cond_dim, code_dim)

        self.up1 = nn.ConvTranspose2d(code_dim, 128, 4, stride=2, padding=1)
        self.res1 = ResBlock(128, 128)

        self.up2 = nn.ConvTranspose2d(128, n_channel, 4, stride=2, padding=1)

    def forward(self, z_q, c):
        z = self.film(z_q, c)
        z = self.res1(self.up1(z))
        return self.up2(z)


class EMA_VQVAE(nn.Module):
    def __init__(self, n_channel, num_codes, code_dim, beta_vq, cond_dim):
        super().__init__()

        self.cond_dim = cond_dim

        self.encoder = ResMapEncoder(n_channel, code_dim)
        self.vq = EMAVectorQuantizer(num_codes, code_dim, beta_vq)


        self.cond_embed = nn.Sequential(
            nn.Linear(self.cond_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )

        self.decoder = ResDecoder(n_channel, code_dim, cond_dim=64)

    def forward(self, x, clip_emb):
        z_e = self.encoder(x)
        z_q, indices, vq_loss = self.vq(z_e)
        c = self.cond_embed(clip_emb)
        logits = self.decoder(z_q, c)
        return logits, indices, vq_loss

    @torch.no_grad()
    def sample(self, clip_emb, indices=None):
        """
        Conditional generation via codebook sampling
        """
        B = clip_emb.size(0)
        device = clip_emb.device

        if indices is None:
            indices = torch.randint(
                0, self.vq.num_codes,
                (B, 4, 4),
                device=device
            )

        # indices → z_q
        z_q = self.vq.codebook(indices.view(-1))
        z_q = z_q.view(B, 4, 4, self.vq.code_dim).permute(0, 3, 1, 2)

        c = self.cond_embed(clip_emb)
        logits = self.decoder(z_q, c)

        return logits