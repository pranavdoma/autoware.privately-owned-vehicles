# The autopad is used to detect the padding value for the Convolution layer
import torch
import torch.nn as nn

def autopad(k, p=None, d=1):
    if d > 1:
        # actual kernel-size
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        # auto-pad
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p
  
# This is the activation function used in YOLOv11
class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

# The base Conv Block

class Conv(torch.nn.Module):

    def __init__(self, in_ch, out_ch, k=1, s=1, p=0, g=1):
        # in_ch = input channels
        # out_ch = output channels
        # activation = the torch function of the activation function (SiLU or Identity)
        # k = kernel size
        # s = stride
        # p = padding
        # g = groups
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.norm = torch.nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.03)
        self.relu = SiLU()
    def forward(self, x):
        # Passing the input by convolution layer and using the activation function
        # on the normalized output
        return self.relu(self.norm(self.conv(x)))
        
    def fuse_forward(self, x):
        return self.relu(self.conv(x))

# The Bottlneck block

class Residual(torch.nn.Module):
    def __init__(self, ch, e=0.5):
        super().__init__()
        self.conv1 = Conv(ch, int(ch * e), k=3, p=1)
        self.conv2 = Conv(int(ch * e), ch, k=3, p=1)

    def forward(self, x):
        # The input is passed through 2 Conv blocks and if the shortcut is true and
        # if input and output channels are same, then it will the input as residual
        return x + self.conv2(self.conv1(x))

# The C3k Module
class C3K(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch // 2)
        self.conv2 = Conv(in_ch, out_ch // 2)
        self.conv3 = Conv(2 * (out_ch // 2), out_ch)
        self.res_m = torch.nn.Sequential(Residual(out_ch // 2, e=1.0),
                                         Residual(out_ch // 2, e=1.0))

    def forward(self, x):
        y = self.res_m(self.conv1(x)) # Process half of the input channels
        # Process the other half directly, Concatenate along the channel dimension
        return self.conv3(torch.cat((y, self.conv2(x)), dim=1))
        
# The C3K2 Module

class C3K2(torch.nn.Module):
    def __init__(self, in_ch, out_ch, n, csp, r):
        super().__init__()
        self.conv1 = Conv(in_ch, 2 * (out_ch // r))
        self.conv2 = Conv((2 + n) * (out_ch // r), out_ch)

        if not csp:
            # Using the CSP Module when mentioned True at shortcut
            self.res_m = torch.nn.ModuleList(Residual(out_ch // r) for _ in range(n))
        else:
            # Using the Bottlenecks when mentioned False at shortcut
            self.res_m = torch.nn.ModuleList(C3K(out_ch // r, out_ch // r) for _ in range(n))

    def forward(self, x):
        y = list(self.conv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.res_m)
        return self.conv2(torch.cat(y, dim=1))
        
 # Code for SPFF Block
class SPPF(nn.Module):

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_          = c1 // 2
        self.cv1    = Conv(c1, c_, 1, 1)
        self.cv2    = Conv(c_ * 4, c2, 1, 1)
        self.m      = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x) # Starting with a Conv Block
        y1 = self.m(x) # First MaxPool layer
        y2 = self.m(y1) # Second MaxPool layer
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1)) # Ending with Conv Block

# Code for the Attention Module

class Attention(nn.Module):
    """Multi-head attention module"""
    def __init__(self, ch, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = ch // num_heads
        self.dim_key = self.dim_head // 2
        self.scale = self.dim_key ** -0.5
        
        self.qkv = Conv(ch, ch + self.dim_key * num_heads * 2, 1, act=False)
        self.conv1 = Conv(ch, ch, k=3, g=ch, act=False)
        self.conv2 = Conv(ch, ch, 1, act=False)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(b, self.num_heads, self.dim_key * 2 + self.dim_head, h * w)
        q, k, v = qkv.split([self.dim_key, self.dim_key, self.dim_head], dim=2)
        
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (v @ attn.transpose(-2, -1)).view(b, c, h, w) + self.conv1(v.reshape(b, c, h, w))
        return self.conv2(x)

# Code for the PSABlock
class PSABlock(nn.Module):
    """PSA attention block"""
    def __init__(self, ch, num_heads=8):
        super().__init__()
        self.conv1 = Attention(ch, num_heads)
        self.conv2 = nn.Sequential(
            Conv(ch, ch * 2),
            Conv(ch * 2, ch, act=False)
        )

    def forward(self, x):
        x = x + self.conv1(x)
        return x + self.conv2(x)

# Code for the C2PSA
class C2PSA(nn.Module):
    """Cross Stage Partial with Spatial Attention"""
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.conv1 = Conv(c1, 2 * c_, 1)
        self.conv2 = Conv(2 * c_, c2, 1)
        num_heads = max(c_ // 64, 1) * 8
        self.res_m = nn.Sequential(*(PSABlock(c_, num_heads) for _ in range(n)))

    def forward(self, x):
        x, y = self.conv1(x).chunk(2, 1)
        return self.conv2(torch.cat((x, self.res_m(y)), 1))

       