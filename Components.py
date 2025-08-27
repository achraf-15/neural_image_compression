import torch.nn as nn
from compressai.layers.gdn import GDN

class Encoder(nn.Module):
    def __init__(self, latent_channels=192, use_gdn=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, latent_channels, kernel_size=5, stride=2, padding=2),
            GDN(latent_channels, beta_min=1e-6, gamma_init=.1) if use_gdn else nn.LeakyReLU(),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=5, stride=2, padding=2),
            GDN(latent_channels, beta_min=1e-6, gamma_init=.1) if use_gdn else nn.LeakyReLU(),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=5, stride=2, padding=2),
            GDN(latent_channels, beta_min=1e-6, gamma_init=.1) if use_gdn else nn.LeakyReLU(),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=5, stride=2, padding=2),  # bottleneck
        )
    def forward(self, x): return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_channels=192, use_gdn=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, latent_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
            GDN(latent_channels, inverse=True, beta_min=1e-6, gamma_init=.1) if use_gdn else nn.LeakyReLU(),
            nn.ConvTranspose2d(latent_channels, latent_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
            GDN(latent_channels, inverse=True, beta_min=1e-6, gamma_init=.1) if use_gdn else nn.LeakyReLU(),
            nn.ConvTranspose2d(latent_channels, latent_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
            GDN(latent_channels, inverse=True, beta_min=1e-6, gamma_init=.1) if use_gdn else nn.LeakyReLU(),
            nn.ConvTranspose2d(latent_channels, 3, kernel_size=5, stride=2, padding=2, output_padding=1),
        )
    def forward(self, y): return self.net(y)

    
class ResidualBlock(nn.Module):
    def __init__(self, latent_channels=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )
    def forward(self, x): return self.net(x) + x


class UpsampleResidualBlock(nn.Module):
    def __init__(self, latent_channels=192, r=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, latent_channels*r, kernel_size=3, stride=r, padding=1, output_padding=r-1),
            nn.LeakyReLU(),
            nn.Conv2d(latent_channels * r, latent_channels * r, kernel_size=3, stride=1, padding=1),
            GDN(latent_channels * r, inverse=True, beta_min=1e-6, gamma_init=.1),
        )
        self.skip = nn.ConvTranspose2d(latent_channels, latent_channels*r, kernel_size=3, stride=r, padding=1, output_padding=r-1)
    def forward(self, x): return self.main(x) + self.skip(x)


class LatentSpaceTransform(nn.Module):
    def __init__(self, latent_channels=192, upsampling_factors=[2,1,1,1]):
        super().__init__()
        self.RB1 = ResidualBlock(latent_channels=latent_channels)
        self.URB1 = UpsampleResidualBlock(latent_channels=latent_channels, r=upsampling_factors[0])
        latent_channels *= upsampling_factors[0]

        self.RB2 = ResidualBlock(latent_channels=latent_channels)
        self.URB2 = UpsampleResidualBlock(latent_channels=latent_channels, r=upsampling_factors[1])
        latent_channels *= upsampling_factors[1]

        self.RB3 = ResidualBlock(latent_channels=latent_channels)
        self.URB3 = UpsampleResidualBlock(latent_channels=latent_channels, r=upsampling_factors[2])
        latent_channels *= upsampling_factors[2]

        self.RB4 = ResidualBlock(latent_channels=latent_channels)
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
        

class HyperEncoder(nn.Module):
    def __init__(self, latent_channels=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=5, stride=2, padding=2),
        )
    def forward(self, y): return self.net(y)

    
class HyperDecoder(nn.Module):
    def __init__(self, latent_channels=192):
        super().__init__()
        # outputs channels ~ 2 * N (paper uses 384) to be concatenated with context features
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, latent_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(latent_channels, int(1.5 * latent_channels), kernel_size=5, stride=2, padding=2, output_padding=1),  # c288
            nn.LeakyReLU(0.2),
            nn.Conv2d(int(1.5 * latent_channels), 2 * latent_channels, kernel_size=3, stride=1, padding=1)  # c384
        )
    def forward(self, z): return self.net(z)

