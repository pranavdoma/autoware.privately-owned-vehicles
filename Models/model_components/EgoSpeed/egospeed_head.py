from egospeed_utils import *
import torch
import torch.nn as nn
class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).
    This version now matches the reference implementation exactly.
    """
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Forward pass for DFL."""
        # Input shape: (batch_size, 4 * reg_max, num_anchor_points)
        bs, _, num_points = x.shape
        # Reshape, transpose, apply softmax, and then the convolution
        # This calculates the weighted sum for the distribution
        return self.conv(x.view(bs, 4, self.c1, num_points).transpose(2, 1).softmax(1)).view(bs, 4, num_points)


class EgoSpeedHead(nn.Module):
    """
    MODIFIED EgoSpeed Head: Anchor-Free and Decoupled.
    This version is modified to follow the anchor-free design pattern.
    It no longer uses pre-defined anchor boxes. Instead, it predicts the
    distances from a grid point to the four sides of a bounding box.
    """
    def __init__(self, nc=3, ch=()):  # nc=3 for CIPO levels 0,1,2
        super(EgoSpeedHead, self).__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels, defines the range for distance prediction
        self.no = nc + self.reg_max * 4  # number of outputs per anchor point

        # Define the strides for each detection layer (P3, P4, P5)
        strides = torch.tensor([8., 16., 32.])
        self.register_buffer('stride', strides)
        
        # Initialize the DFL module, which is essential for decoding the box predictions
        self.dfl = DFL(self.reg_max)

        # Decoupled head: Separate convolutional modules for box and class prediction
        self.box_preds = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        for i in ch:
            self.box_preds.append(
                nn.Sequential(
                    Conv(i, i, k=3),
                    Conv(i, i, k=3),
                    nn.Conv2d(i, 4 * self.reg_max, 1)
                )
            )
            self.cls_preds.append(
                nn.Sequential(
                    Conv(i, i, k=3),
                    Conv(i, i, k=3),
                    nn.Conv2d(i, self.nc, 1)
                )
            )

    def forward(self, x):
        """Forward pass through the anchor-free head."""
        for i in range(self.nl):
            # Pass the input through the separate box and class prediction branches
            x[i] = torch.cat((self.box_preds[i](x[i]), self.cls_preds[i](x[i])), 1)

        if self.training:
            return x

        # --- INFERENCE MODE ---
        # Decode the raw output into final bounding boxes
        bs = x[0].shape[0]
        # Create the grid of anchor points and the corresponding strides
        anchor_points, stride_tensor = self._make_anchor_points_and_strides(x)

        # Concatenate and reshape the outputs from all detection layers
        x_cat = torch.cat([xi.view(bs, self.no, -1) for xi in x], 2)
        box_dist, cls_out = x_cat.split((self.reg_max * 4, self.nc), 1)
        
        # Use DFL to decode the box distribution into ltrb (left, top, right, bottom) distances
        box_ltrb = self.dfl(box_dist) * stride_tensor
        
        # Calculate the final box coordinates (x1, y1, x2, y2)
        x1y1 = anchor_points - box_ltrb[:, :2] # top-left
        x2y2 = anchor_points + box_ltrb[:, 2:] # bottom-right
        
        # Combine box coordinates and apply sigmoid to class predictions for final output
        return torch.cat((x1y1, x2y2, cls_out.sigmoid()), 1)
    
    def _make_anchor_points_and_strides(self, features):
        """Generates a grid of anchor points and their strides for all feature maps."""
        anchor_points, stride_tensor = [], []
        dtype, device = features[0].dtype, features[0].device
        for i, stride in enumerate(self.stride):
            _, _, h, w = features[i].shape
            # Create a grid of (x, y) coordinates for this feature map
            sx = torch.arange(w, device=device, dtype=dtype) + 0.5
            sy = torch.arange(h, device=device, dtype=dtype) + 0.5
            sy, sx = torch.meshgrid(sy, sx, indexing='ij')
            
            anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w,), stride, device=device, dtype=dtype))
        
        # Concatenate the points and strides from all layers
        return torch.cat(anchor_points).transpose(0, 1), torch.cat(stride_tensor)

