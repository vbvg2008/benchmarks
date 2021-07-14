import pdb

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class ViTEmbeddings(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_channels=3, em_dim=768, drop=0.1) -> None:
        super().__init__()
        assert image_size % patch_size == 0, "image size must be an integer multiply of patch size"
        self.patch_embedding = nn.Conv2d(num_channels, em_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.position_embedding = nn.Parameter(torch.zeros(1, (image_size // patch_size)**2 + 1, em_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, em_dim))
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x = self.patch_embedding(x).flatten(2).transpose(1, 2)  # [B,C, H, W] -> [B, num_patches, em_dim]
        x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1)  # [B, num_patches+1, em_dim]
        x = x + self.position_embedding
        x = self.dropout(x)
        return x


class ViTModel(nn.Module):
    def __init__(self,
                 num_classes,
                 num_layers=12,
                 image_size=224,
                 patch_size=16,
                 num_channels=3,
                 em_dim=768,
                 drop=0.0,
                 num_heads=12,
                 ff_dim=3072):
        super().__init__()
        self.embedding = ViTEmbeddings(image_size, patch_size, num_channels, em_dim, drop)
        encoder_layer = TransformerEncoderLayer(em_dim,
                                                nhead=num_heads,
                                                dim_feedforward=ff_dim,
                                                activation='gelu',
                                                dropout=drop)
        self.encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.layernorm = nn.LayerNorm(em_dim, eps=1e-6)
        self.classifier = nn.Linear(em_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(0, 1)  # Switch batch and sequence length dimension for pytorch convention
        x = self.encoder(x)
        x = self.layernorm(x[0])
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = ViTModel(num_classes=10, num_layers=4, em_dim=256, num_heads=8, ff_dim=512)
    inputs = torch.randn(8, 3, 224, 224)
    outputs = model(inputs)
    pdb.set_trace()
