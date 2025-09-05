import torch
import torch.nn as nn   

# ============================================
# Base Components (from provided utils)
# ============================================

def autopad(k, p=None, d=1):
    """Auto-calculate padding for Conv layer"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class SiLU(nn.Module):
    """SiLU activation function"""
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

class Conv(nn.Module):
    """Standard convolution with batch norm and activation"""
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        p = autopad(k, p)
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.norm = nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.03)
        self.relu = SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

class Residual(nn.Module):
    """Residual block for C3k2"""
    def __init__(self, ch, e=0.5):
        super().__init__()
        c_ = int(ch * e)
        self.conv1 = Conv(ch, c_, k=3)
        self.conv2 = Conv(c_, ch, k=3)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))

class C3k2(nn.Module):
    """C3k2 module with CSP bottleneck"""
    def __init__(self, in_ch, out_ch, n=1, shortcut=False, e=0.5):
        super().__init__()
        c_ = int(out_ch * e)
        self.conv1 = Conv(in_ch, 2 * c_, 1)
        self.conv2 = Conv((2 + n) * c_, out_ch, 1)
        self.res_m = nn.ModuleList(Residual(c_) for _ in range(n))

    def forward(self, x):
        y = list(self.conv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.res_m)
        return self.conv2(torch.cat(y, 1))

class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast"""
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

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

class Concat(nn.Module):
    """Concatenate tensors along dimension"""
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)

class Upsample(nn.Module):
    """Upsample using nearest neighbor"""
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale_factor, mode='nearest')