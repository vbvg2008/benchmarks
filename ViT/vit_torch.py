import pdb

import torch
import torch.nn as nn


class ViTEmbeddings(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_channels=3, embed_dim=768, drop=0.1) -> None:
        super().__init__()
        assert image_size % patch_size == 0, "image size must be an integer multiply of patch size"
        self.patch_embedding = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.position_embedding = nn.Parameter(torch.zeros(1, (image_size // patch_size)**2 + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x = self.patch_embedding(x).flatten(2).transpose(1, 2)  # [B,C, H, W] -> [B, num_patches, embed_dim]
        x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1)  # [B, num_patches+1, embed_dim]
        x = x + self.position_embedding
        x = self.dropout(x)
        return x


# class ViTModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.embedding = ViTEmbeddings()

if __name__ == "__main__":
    model = ViTEmbeddings()
    inputs = torch.randn(8, 3, 224, 224)
    outputs = model(inputs)
    pdb.set_trace()
