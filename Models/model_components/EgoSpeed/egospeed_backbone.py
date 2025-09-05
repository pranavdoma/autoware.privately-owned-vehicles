from .egospeed_utils import *
import torch

class EgoSpeedBackbone(nn.Module):
    """
    EgoSpeed Backbone
    """
    def __init__(self, nc=3, anchors=None):
        super().__init__()
        # self.nc = nc  # number of CIPO classes
        # self.anchors = anchors
        self.p1 =[]
        self.p2 =[]
        self.p3 =[]
        self.p4 =[]
        self.p5 =[]

        # ==================== BACKBONE ====================
        # Initial downsampling
        self.conv0 = Conv(3, 32, k=3, s=2)  # 640->320, P1/2
        self.conv1 = Conv(32, 64, k=3, s=2)  # 320->160, P2/4
        
        # Stage 1
        self.c3k2_1 = C3k2(64, 128, n=1, shortcut=True)  # P3/8
        self.conv2 = Conv(128, 128, k=3, s=2)  # 160->80
        
        # Stage 2 
        self.c3k2_2 = C3k2(128, 256, n=2, shortcut=True)  # P4/16
        self.conv3 = Conv(256, 256, k=3, s=2)  # 80->40
        
        # Stage 3
        self.c3k2_3 = C3k2(256, 512, n=2, shortcut=True)  # P5/32
        self.conv4 = Conv(512, 512, k=3, s=2)  # 40->20
        
        # Stage 4
        self.c3k2_4 = C3k2(512, 512, n=1, shortcut=True)
        
        # SPPF and C2PSA
        self.sppf = SPPF(512, 512, k=5)
        self.c2psa = C2PSA(512, 512, n=1)

    def forward(self, x):
        """
        Forward pass for backbone return p3 ,p4 ,p5
        """
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.c3k2_1(x)
        x = self.conv2(x)
        x = self.c3k2_2(x)
        p3 = x
        x = self.conv3(x)
        x = self.c3k2_3(x)
        p4= x
        x = self.conv4(x)
        x = self.c3k2_4(x)
        x = self.sppf(x)
        x = self.c2psa(x)
        p5 = x
        return p3, p4, p5   