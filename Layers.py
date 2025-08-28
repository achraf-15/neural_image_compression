import torch.nn as nn
from torch import Tensor
from compressai.layers.gdn import GDN


class SubpelConv3x3(nn.Module):
    def __init__(self, in_ch, out_ch, upsample=2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch * (upsample ** 2),
            kernel_size=3, stride=1, padding=1
        )
        self.shuffle = nn.PixelShuffle(upsample)

    def forward(self, x):
        return self.shuffle(self.conv(x))

class TransposedDeconv3x3(nn.Module):
    def __init__(self, in_ch, out_ch, upsample=2):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=upsample, padding=1, output_padding=upsample-1)

    def forward(self, x):
        return self.deconv(x)


class ResidualBlockWithStride(nn.Module):
    """Residual block with a stride on the first convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.gdn = GDN(out_ch, beta_min=1e-6, gamma_init=.1)
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)
        else:
            self.skip = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.gdn(out)

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        return out


class ResidualBlockUpsample(nn.Module):
    """Residual block with sub-pixel upsampling on the last convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        upsample (int): upsampling factor (default: 2)
    """

    def __init__(self, in_ch: int, out_ch: int, upsample: int = 2):
        super().__init__()
        self.subpel_conv = TransposedDeconv3x3(in_ch, out_ch, upsample) # SubpelConv3x3
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv =  nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.igdn = GDN(out_ch, inverse=True, beta_min=1e-6, gamma_init=.1)
        self.upsample = TransposedDeconv3x3(in_ch, out_ch, upsample) # SubpelConv3x3

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.subpel_conv(x)
        out = self.leaky_relu(out)
        out = self.conv(out)
        out = self.igdn(out)
        identity = self.upsample(x)
        out += identity
        return out


class ResidualBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1)
        else:
            self.skip = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return out