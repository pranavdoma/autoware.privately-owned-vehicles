from .egospeed_utils import *
import torch
import torch.nn as nn

class EgoSpeedHead(nn.Module):
    """
    EgoSpeed Head
    """
    def __init__(self, nc=3, ch=()):  # nc=3 for CIPO levels 0,1,2
        super(EgoSpeedHead, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor (x,y,w,h,obj,c1,c2,c3)
        self.nl = len(ch)  # number of detection layers
        self.na = 3  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl
        self.anchor_grid = [torch.zeros(1)] * self.nl
        self.stride = torch.zeros(self.nl)
        
        # Detection head convolutions
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, x, k=3, s=1),
                Conv(x, x, k=3, s=1),
                Conv(x, self.na * self.no, k=1, s=1)
            ) for x in ch
        )
        
    def forward(self, x):
        """Forward pass through detection head"""
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.cv2[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            
            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                
                xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))
        
        return x if self.training else (torch.cat(z, 1), x)
    
    def _make_grid(self, nx=20, ny=20, i=0):
        """Create mesh grid for predictions"""
        d = self.anchors[i].device
        yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]).view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid
