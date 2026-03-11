import torch
from torch import nn
import timm
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPModel


class SqueezeExcite(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, h, w = x.shape
        y = x.mean(dim=(2,3))
        y = self.fc1(y)
        y = F.gelu(y)
        y = self.fc2(y)
        y = torch.sigmoid(y)
        y = y.view(b, c, 1, 1)
        x = x * y
        return x


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, drop_rate=0.0, use_se=False):
        super().__init__()
        self.use_se = use_se
        self.drop_rate = drop_rate

        self.dw = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.norm = nn.LayerNorm(in_ch)

        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)

        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        else:
            self.shortcut = nn.Identity()

        self.dropout = nn.Dropout(drop_rate)

        if use_se:
            self.se = SqueezeExcite(out_ch)
        else:
            self.se = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        x = self.dw(x)
        x = F.gelu(x)

        x = x.permute(0,2,3,1)
        x = self.norm(x)
        x = x.permute(0,3,1,2)

        x = self.pw(x)
        x = self.se(x)
        x = self.dropout(x)

        return x + residual



class CNNResMapEncoder(nn.Module):
    def __init__(self, num_classes, embedding_dim, drop_rate):
        super().__init__()

        self.conv1 = nn.Conv2d(num_classes, 64, kernel_size=3, padding=1)
        self.norm1 = nn.LayerNorm(64)

        self.res1 = ResBlock(64, 128, drop_rate=drop_rate, use_se=True)

        self.down = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.LayerNorm(128)

        self.res2 = ResBlock(128, 256, drop_rate=drop_rate, use_se=True)

        self.fc1 = nn.Linear(256, 256)
        self.norm3 = nn.LayerNorm(256)
        self.dropout = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(256, embedding_dim, bias=False)

        self.embedding_dim = embedding_dim

    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(0,2,3,1)
        x = self.norm1(x)
        x = x.permute(0,3,1,2)
        x = F.gelu(x)

        x = self.res1(x)

        x = self.down(x)
        x = x.permute(0,2,3,1)
        x = self.norm2(x)
        x = x.permute(0,3,1,2)
        x = F.gelu(x)

        x = self.res2(x)

        x = x.mean(dim=(2,3))

        x = self.fc1(x)
        x = F.gelu(x)
        x = self.norm3(x)
        x = self.dropout(x)

        x = self.fc2(x)
        return x


class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class LevelEncoder(nn.Module):
    def __init__(self, num_classes, embedding_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(num_classes, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),

            nn.Flatten(),

            MLPHead(128, 1024, embedding_dim)
        )
        self.embedding_dim = embedding_dim

    def forward(self, x):
        x = self.encoder(x)
        return x
        

class TextEncoder(nn.Module):
    def __init__(self, text_encoder_model, embedding_dim):
        super().__init__()
        self.text_model = CLIPModel.from_pretrained(text_encoder_model)
        self.projection = nn.Linear(512, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, input_ids, attention_mask):
        x = self.text_model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).detach()
        x = self.projection(x)

        return x

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Override state_dict to exclude self.text_model (pretrained CLIP)"""
        state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        # Filter out text_model parameters
        keys_to_remove = [k for k in state_dict.keys() if k.startswith(prefix + 'text_model.')]
        for key in keys_to_remove:
            del state_dict[key]
        return state_dict

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Override to allow loading without text_model parameters"""
        # The pretrained text_model will be loaded from HuggingFace, not from checkpoint
        super()._load_from_state_dict(state_dict, prefix, local_metadata, False, missing_keys, unexpected_keys, error_msgs)
