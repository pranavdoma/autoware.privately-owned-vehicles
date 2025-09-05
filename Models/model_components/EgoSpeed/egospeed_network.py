from egospeed_backbone import EgoSpeedBackbone
from egospeed_neck import EgoSpeedNeck
from egospeed_head import EgoSpeedHead
import torch.nn as nn
import torch


class EgoSpeedNetwork(nn.Module):
    """
    EgoSpeed Network
    """
    def __init__(self, nc=3, ch=()):
        super(EgoSpeedNetwork, self).__init__()
        self.backbone = EgoSpeedBackbone(nc, ch)
        self.neck = EgoSpeedNeck(nc, ch)
        self.head = EgoSpeedHead(nc, ch)

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        p3, p4, p5 = self.neck(p3, p4, p5)
        return self.head([p3, p4, p5])