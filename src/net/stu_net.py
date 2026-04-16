import timm
from torch import nn as nn


# 模型定义
class DistillationStudent(nn.Module):
    def __init__(self, target_dim=256, teacher_dim=1024):
        super().__init__()
        self.backbone = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=0)
        backbone_dim = self.backbone.num_features

        self.retrieval_head = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, target_dim)
        )

        self.distill_head = nn.Sequential(
            nn.Linear(target_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, teacher_dim)
        )

    def forward(self, x):
        features = self.backbone(x)
        embed_256 = self.retrieval_head(features)
        if self.training:
            embed_1024 = self.distill_head(embed_256)
            return embed_256, embed_1024
        return embed_256
