from egospeed_utils import *
import torch
import torch.nn as nn

class EgoSpeedNeck(nn.Module):
    """
    EgoSpeed Neck
    """
    def __init__(self, in_channels, out_channels):
        super(EgoSpeedNeck, self).__init__()
        # in_channels and out_channels are tuples of channel sizes
        self.upsample1 =  torch.nn.Upsample(scale_factor=2)
        self.c3k2_5 = C3k2(512 + 512, 512, n=1, shortcut=False)

        self.upsample2 =  torch.nn.Upsample(scale_factor=2)
        self.c3k2_6 = C3k2(512 + 256, 256, n=1, shortcut=False)

        # Bottom-up pathway
        self.conv5 = Conv(256, 256, k=3, s=2)
        self.c3k2_7 = C3k2(256 + 512, 512, n=1, shortcut=False)
        
        self.conv6 = Conv(512, 512, k=3, s=2)
        self.c3k2_8 = C3k2(512 + 512, 512, n=1, shortcut=False)

    def forward(self, p3, p4, p5):
        x = self.upsample1(p5)
        x= torch.cat((x, p4), 1)
        x = self.c3k2_5(x)
        p4 = x

        x = self.upsample2(p4)
        x = torch.cat((x, p3), 1)
        x = self.c3k2_6(x)
        p3 = x

        x = self.conv5(p3)
        x = torch.cat((x, p4), 1)
        x = self.c3k2_7(x)
        p4 = x

        x = self.conv6(x)
        x = torch.cat((x, p5), 1)
        x = self.c3k2_8(x)
        p5 = x

        return p3, p4, p5
    

    


       
        