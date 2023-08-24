"""by lyuwenyu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np

from src.core import register

__all__ = ['RTDETR', ]


# 主模型,很少的代码
@register
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale

    def forward(self, x, targets=None):
        # todo
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])
        # 经过backbone
        x = self.backbone(x)
        # 经过encoder
        x = self.encoder(x)
        # 经过decoder
        x = self.decoder(x, targets)

        return x

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
