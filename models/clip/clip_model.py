import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from models.clip.encoders import LevelEncoder, TextEncoder, CNNResMapEncoder


class CLIPModel(nn.Module):
    def __init__(
        self,
        num_classes,
        embedding_dim,
        drop_rate,
        init_temperature,
        text_encoder_model
    ):
        super().__init__()
        self.level_encoder = CNNResMapEncoder(num_classes, embedding_dim, drop_rate) # or LevelEncoder
        self.text_encoder = TextEncoder(text_encoder_model, embedding_dim)
        
        self.temperature = nn.Parameter(
            torch.tensor(np.log(init_temperature), dtype=torch.float32)
        )

    def forward(self, batch):
        level_embeddings = self.level_encoder(batch["level"])
        text_embeddings = self.text_encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        
        level_embeddings = F.normalize(level_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)

        return level_embeddings, text_embeddings


class CLIPFrozenModel(nn.Module):
    def __init__(
        self,
        embedding_dim,
        text_encoder_model
    ):
        super().__init__()
        self.text_encoder = TextEncoder(text_encoder_model, embedding_dim)

        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def forward(self, batch):
        level_embeddings = self.level_encoder(batch["level"])
        with torch.no_grad():
            text_embeddings = self.text_encoder(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )

        level_embeddings = F.normalize(level_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)

        return level_embeddings, text_embeddings