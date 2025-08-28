import torch.nn as nn
from compressai.layers.gdn import GDN

from Layers import ResidualBlock, ResidualBlockUpsample, ResidualBlockWithStride, SubpelConv3x3, TransposedDeconv3x3

class Encoder5x5(nn.Module):
    def __init__(self, latent_channels=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, latent_channels, kernel_size=5, stride=2, padding=2),
            GDN(latent_channels, beta_min=1e-6, gamma_init=.1),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=5, stride=2, padding=2),
            GDN(latent_channels, beta_min=1e-6, gamma_init=.1),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=5, stride=2, padding=2),
            GDN(latent_channels, beta_min=1e-6, gamma_init=.1),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=5, stride=2, padding=2),  # bottleneck
        )
    def forward(self, x): return self.net(x)

class Encoder3x3(nn.Module):
    def __init__(self, latent_channels=192):
        super().__init__()
        self.net = nn.Sequential(
            ResidualBlockWithStride(3, latent_channels, stride=2),
            ResidualBlock(latent_channels, latent_channels),
            ResidualBlockWithStride(latent_channels, latent_channels, stride=2),
            ResidualBlock(latent_channels, latent_channels),
            ResidualBlockWithStride(latent_channels, latent_channels, stride=2),
            ResidualBlock(latent_channels, latent_channels),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, stride=2, padding=1),  # bottleneck
        )
    def forward(self, x): return self.net(x)


class Decoder5x5(nn.Module):
    def __init__(self, latent_channels=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, latent_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
            GDN(latent_channels, inverse=True, beta_min=1e-6, gamma_init=.1),
            nn.ConvTranspose2d(latent_channels, latent_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
            GDN(latent_channels, inverse=True, beta_min=1e-6, gamma_init=.1),
            nn.ConvTranspose2d(latent_channels, latent_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
            GDN(latent_channels, inverse=True, beta_min=1e-6, gamma_init=.1),
            nn.ConvTranspose2d(latent_channels, 3, kernel_size=5, stride=2, padding=2, output_padding=1),
        )
    def forward(self, y): return self.net(y)

class Decoder3x3(nn.Module):
    def __init__(self, latent_channels=192):
        super().__init__()
        self.net = nn.Sequential(
            ResidualBlock(latent_channels, latent_channels),
            ResidualBlockUpsample(latent_channels, latent_channels, 2),
            ResidualBlock(latent_channels, latent_channels),
            ResidualBlockUpsample(latent_channels, latent_channels, 2),
            ResidualBlock(latent_channels, latent_channels),
            ResidualBlockUpsample(latent_channels, latent_channels, 2),
            ResidualBlock(latent_channels, latent_channels),
            TransposedDeconv3x3(latent_channels, 3, 2), # SubpelConv3x3
        )
    def forward(self, y): return self.net(y)

        
class HyperEncoder5x5(nn.Module):
    def __init__(self, latent_channels=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=5, stride=2, padding=2),
        )
    def forward(self, y): return self.net(y)

class HyperEncoder3x3(nn.Module):
    def __init__(self, latent_channels=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, stride=2, padding=1),
        )
    def forward(self, y): return self.net(y)

    
class HyperDecoder5x5(nn.Module):
    def __init__(self, latent_channels=192):
        super().__init__()
        # outputs channels ~ 2 * N (paper uses 384) to be concatenated with context features
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, latent_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(latent_channels, int(1.5 * latent_channels), kernel_size=5, stride=2, padding=2, output_padding=1),  # c288
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(int(1.5 * latent_channels), 2 * latent_channels, kernel_size=3, stride=1, padding=1)  # c384
        )
    def forward(self, z): return self.net(z)

class HyperDecoder3x3(nn.Module):
    def __init__(self, latent_channels=192):
        super().__init__()
        # outputs channels ~ 2 * N (paper uses 384) to be concatenated with context features
        self.net = nn.Sequential(
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            TransposedDeconv3x3(latent_channels, latent_channels, 2), # SubpelConv3x3
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(latent_channels, int(1.5 * latent_channels), kernel_size=3, stride=1, padding=1),  # c288
            nn.LeakyReLU(inplace=True),
            TransposedDeconv3x3(int(1.5 * latent_channels), int(1.5 * latent_channels), 2), # SubpelConv3x3
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(int(1.5 * latent_channels), 2 * latent_channels, kernel_size=3, stride=1, padding=1)  # c384
        )
    def forward(self, z): return self.net(z)


class LatentSpaceTransform(nn.Module):
    def __init__(self, latent_channels=192, upsampling_factors=[2,1,1,1]):
        super().__init__()
        self.RB1 = ResidualBlock(in_ch=latent_channels, out_ch=latent_channels)
        self.URB1 = ResidualBlockUpsample(in_ch=latent_channels, out_ch=latent_channels, upsample=upsampling_factors[0])
        latent_channels *= upsampling_factors[0]

        self.RB2 = ResidualBlock(in_ch=latent_channels, out_ch=latent_channels)
        self.URB2 = ResidualBlockUpsample(in_ch=latent_channels, out_ch=latent_channels, upsample=upsampling_factors[1])
        latent_channels *= upsampling_factors[1]

        self.RB3 = ResidualBlock(in_ch=latent_channels, out_ch=latent_channels)
        self.URB3 = ResidualBlockUpsample(in_ch=latent_channels, out_ch=latent_channels, upsample=upsampling_factors[2])
        latent_channels *= upsampling_factors[2]

        self.RB4 = ResidualBlock(in_ch=latent_channels, out_ch=latent_channels)
        self.conv = nn.Conv2d(latent_channels, latent_channels*upsampling_factors[3], kernel_size=3, stride=1, padding=1)

    def forward(self, x): 
        x = self.RB1(x)
        x = self.URB1(x)
        x = self.RB2(x)
        x = self.URB2(x)
        x = self.RB3(x)
        x = self.URB3(x)
        x = self.RB4(x)
        x = self.conv(x)

        return x