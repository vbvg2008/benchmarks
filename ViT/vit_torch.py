import pdb
import tempfile

import fastestimator as fe
import torch
import torch.nn as nn
from fastestimator.dataset.data.cifair100 import load_data
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, PadIfNeeded, RandomCrop
from fastestimator.op.numpyop.univariate import ChannelTranspose, CoarseDropout, Normalize
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.schedule import cosine_decay
from fastestimator.trace.adapt import LRScheduler
from fastestimator.trace.io import BestModelSaver, RestoreWizard
from fastestimator.trace.metric import Accuracy
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
                 drop=0.1,
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
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(0, 1)  # Switch batch and sequence length dimension for pytorch convention
        x = self.encoder(x)
        x = self.layernorm(x[0])
        x = self.dropout(x)
        x = self.classifier(x)
        return x


def get_estimator(epochs=200,
                  batch_size=128,
                  patch_size=4,
                  model_dir=tempfile.mkdtemp(),
                  restore_dir=tempfile.mkdtemp()):
    # step 1: prepare dataset
    train_data, eval_data = load_data()
    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        batch_size=batch_size,
        ops=[
            Normalize(inputs="x", outputs="x", mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
            PadIfNeeded(min_height=40, min_width=40, image_in="x", image_out="x", mode="train"),
            RandomCrop(32, 32, image_in="x", image_out="x", mode="train"),
            Sometimes(HorizontalFlip(image_in="x", image_out="x", mode="train")),
            CoarseDropout(inputs="x", outputs="x", mode="train", max_holes=1),
            ChannelTranspose(inputs="x", outputs="x")
        ])
    model = fe.build(model_fn=lambda: ViTModel(num_classes=100, image_size=32, patch_size=patch_size),
                     optimizer_fn=lambda x: torch.optim.SGD(x, lr=0.01, momentum=0.9, weight_decay=1e-4))
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce", from_logits=True),
        UpdateOp(model=model, loss_name="ce")
    ])
    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
        BestModelSaver(model=model, save_dir=model_dir, metric="accuracy", save_best_mode="max"),
        LRScheduler(model=model, lr_fn=lambda epoch: cosine_decay(epoch, cycle_length=epochs, init_lr=0.01)),
        RestoreWizard(directory=restore_dir)
    ]
    estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=epochs, traces=traces)
    return estimator
