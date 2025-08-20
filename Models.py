import math
import torch
import torch.nn as nn
import torch.optim as optim
from compressai.layers.gdn import GDN
from typing import Tuple

from EntropyModels import FactorizedEntropyBottleneck, EntropyParameters, discretized_gaussian_pmf, discretized_mixture_pmf


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


class MaskedConv2d(nn.Conv2d): # taken from Github 
    """
    Implementation of the Masked convolution from the paper
    Van den Oord, Aaron, et al. "Conditional image generation with pixelcnn decoders."
    https://arxiv.org/pdf/1606.05328.pdf
    """
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in ('A', 'B')
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class ContextModel(nn.Module): # taken from Github 
    def __init__(self, latent_channels=192):
        super(ContextModel, self).__init__()
        self.masked = MaskedConv2d(
            "A",
            in_channels=latent_channels,
            out_channels=2 * latent_channels,
            kernel_size=5,
            stride=1,
            padding=2
        )

    def forward(self, x):
        return self.masked(x)



class JointAutoregressiveHierarchical(nn.Module):
    """
    latent_channels : int, default=192, Number of channels in bottleneck y (M).
    K : int, default=1, Number of mixture components. 
        If K=1 → Mean-Scale Gaussian.
        If K>1 → Mixture of Gaussians.
    use_gdn : bool, default=True, Whether to use GDN activations in encoder/decoder.
    """
    def __init__(self,
                 latent_channels: int = 192,
                 K: int = 1, 
                 use_gdn = True
                ):
        super().__init__()

        if not isinstance(latent_channels, int) or latent_channels < 1:
            raise ValueError(f"latent_channels must be int >= 1, got {latent_channels}")
        if not isinstance(K, int) or K < 1:
            raise ValueError(f"K must be int >= 1, got {K}")
        
        self.M = latent_channels
        self.K = K
        self.H = latent_channels # hyper_latents
        self.distribution = 'Mean-Scale Gaussian' if K == 1 else 'Mixture of Gaussians'
        
        self.encoder = Encoder(latent_channels=self.M, use_gdn=use_gdn)
        self.decoder = Decoder(latent_channels=self.M, use_gdn=use_gdn)
        self.hyper_encoder = HyperEncoder(latent_channels=self.M)
        self.hyper_decoder = HyperDecoder(latent_channels=self.M)
        self.factorized_entropy_model = FactorizedEntropyBottleneck(self.M)
        # context model: takes y (M channels) -> context_out_channels (e.g. 2*M)
        self.context_model = ContextModel(latent_channels=self.M)
        # entropy params: takes concat(hyper_out, context_out) (4*M) -> mu, scale (2*M), or weights, mus, scales
        self.entropy_parameters = EntropyParameters(
            latent_channels=self.M,
            hyper_latent_channels=self.H,
            K=self.K
        )
         

    def forward(self, x: torch.Tensor, training: bool = True, debug=False):

        # analysis transform
        y = self.encoder(x)      # shape [B, M, Hy, Wy]
        z = self.hyper_encoder(y)  # shape [B, M, Hz, Wz]

        if training:
            # additive uniform noise relaxation
            z_tilde = z + (torch.rand_like(z) - 0.5)
            y_tilde = y + (torch.rand_like(y) - 0.5)
            z_in = z_tilde
            y_in = y_tilde
        else:
            # inference: quantize
            z_q = torch.round(z)
            y_q = torch.round(y)
            z_in = z_q
            y_in = y_q

        # hyper synthesis -> hyper features
        psi = self.hyper_decoder(z_in)  # shape [B, 2*M, H', W']
        # context model using masked convs: training uses relaxed y_tilde (parallel masked conv)
        phi = self.context_model(y_in)  # shape [B, 2*M, H', W']
        # combine features and predict mu and sigma for y
        combined = torch.cat([phi, psi], dim=1) # shape [B, 4*M, H', W']
        
        if self.distribution == 'Mean-Scale Gaussian':
            mu, sigma = self.entropy_parameters(combined)  # each [B, M, H', W']
        if self.distribution == 'Mixture of Gaussians':
            weights, mus, sigmas = self.entropy_parameters(combined) # each [B, K, M, H', W']
        
        # likelihoods:
        p_z = self.factorized_entropy_model(z_in, debug) # [B, M, Hz, Wz], probabilities (training: z_tilde)
        logp_z = torch.log(p_z)
        
        if self.distribution == 'Mean-Scale Gaussian':
            # y: discretized gaussian mass under mu, sigma evaluated at y_in
            p_y = discretized_gaussian_pmf(y_in, mu, sigma)  
        if self.distribution == 'Mixture of Gaussians':
            p_y = discretized_mixture_pmf(y_in, weights, mus, sigmas)
        logp_y = torch.log(p_y) # [B, M, Hy, Wy], probabilities (training: y_tilde)
           

        # synthesis / reconstruction uses relaxed or quantized y
        x_hat = self.decoder(y_in)

        out = {
            'x_hat': x_hat,
            'y': y,
            'y_in': y_in,          # y_tilde (train) or y_q (eval)
            'z': z,
            'z_in': z_in,          # z_tilde (train) or z_q (eval)
            'p_z': p_z,
            'logp_z': logp_z,
            'p_y': p_y,
            'logp_y': logp_y,
            'training': training,
        }

        if self.distribution == 'Mean-Scale Gaussian':
            out.update({'mu': mu, 'sigma': sigma})
        if self.distribution == 'Mixture of Gaussians':
            out.update({'weights': weights, 'mus': mus, 'sigmas': sigmas})
        
        return out



class ScalableImageCoding(nn.Module):
    """
    latent_channels : int, default=192, Number of channels in bottleneck y (M).
    base_channels: int, default=128, Number of base channels y1 (M1).
    K : int, default=1, Number of mixture components. 
        If K=1 → Mean-Scale Gaussian.
        If K>1 → Mixture of Gaussians.
    use_gdn : bool, default=True, Whether to use GDN activations in encoder/decoder.
    """
    def __init__(self,
                 latent_channels: int = 192,
                 base_channels: int= 128,
                 K: int = 1, 
                 use_gdn = True
                ):
        super().__init__()

        if not isinstance(latent_channels, int) or latent_channels < 1:
            raise ValueError(f"latent_channels must be int >= 1, got {latent_channels}")
        if not isinstance(K, int) or K < 1:
            raise ValueError(f"K must be int >= 1, got {K}")
        
        self.M = latent_channels
        self.M1 = base_channels
        self.M2 = latent_channels - base_channels
        self.H = latent_channels # hyper-latents
        self.K = K
        self.distribution = 'Mean-Scale Gaussian' if K == 1 else 'Mixture of Gaussians'
        
        self.encoder = Encoder(latent_channels=self.M, use_gdn=use_gdn)
        self.decoder = Decoder(latent_channels=self.M, use_gdn=use_gdn)
        self.hyper_encoder = HyperEncoder(latent_channels=self.M)
        self.hyper_decoder = HyperDecoder(latent_channels=self.M)
        self.factorized_entropy_model = FactorizedEntropyBottleneck(self.M)
        # context model: takes y (M channels) -> context_out_channels (e.g. 2*M)
        self.context_model_1 = ContextModel(latent_channels=self.M1)
        self.context_model_2 = ContextModel(latent_channels=self.M2)
        # entropy params: takes concat(hyper_out, context_out) (4*M) -> mu, scale (2*M), or weights, mus, scales
        self.entropy_parameters_1 = EntropyParameters(
            latent_channels=self.M1,
            hyper_latent_channels=self.H,
            K=self.K
        )
        self.entropy_parameters_2 = EntropyParameters(
            latent_channels=self.M2,
            hyper_latent_channels=self.H,
            K=self.K
        )
        # Latent Space Transform
        self.LST = LatentSpaceTransform(latent_channels=self.M1, upsampling_factors=[2,1,1,1])
         

    def forward(self, x: torch.Tensor, training: bool = True, debug=False):

        # analysis transform
        y = self.encoder(x)      # shape [B, M, Hy, Wy]
        z = self.hyper_encoder(y)  # shape [B, M, Hz, Wz]

        if training:
            # additive uniform noise relaxation
            z_tilde = z + (torch.rand_like(z) - 0.5)
            y_tilde = y + (torch.rand_like(y) - 0.5)
            z_in = z_tilde
            y_in = y_tilde
        else:
            # inference: quantize
            z_q = torch.round(z)
            y_q = torch.round(y)
            z_in = z_q
            y_in = y_q

        # Split y_in to y1 and y2 
        y1, y2  = torch.split(y_in, [self.M1, self.M2], dim=1)

        # hyper synthesis -> hyper features
        psi = self.hyper_decoder(z_in)  # shape [B, 2*M, H', W']
        # context model using masked convs: training uses relaxed y_tilde (parallel masked conv)
        phi1 = self.context_model_1(y1)  # shape [B, 2*M1, H', W']
        phi2 = self.context_model_2(y2)  # shape [B, 2*M2, H', W']
        # combine features and predict mu and sigma for y
        combined1 = torch.cat([phi1, psi], dim=1) # shape [B, 2*M+M1, H', W']
        combined2 = torch.cat([phi2, psi], dim=1) # shape [B, 2*M+M2, H', W']
        
        if self.distribution == 'Mean-Scale Gaussian':
            mu1, sigma1 = self.entropy_parameters_1(combined1)  # each [B, M, H', W']
            mu2, sigma2 = self.entropy_parameters_2(combined2)  # each [B, M, H', W']
        if self.distribution == 'Mixture of Gaussians':
            weights1, mus1, sigmas1 = self.entropy_parameters_1(combined1) # each [B, K, M, H', W']
            weights2, mus2, sigmas2 = self.entropy_parameters_2(combined2) # each [B, K, M, H', W']
        
        # likelihoods:
        p_z = self.factorized_entropy_model(z_in, debug) # [B, M, Hz, Wz], probabilities (training: z_tilde)
        logp_z = torch.log(p_z)
        
        if self.distribution == 'Mean-Scale Gaussian':
            # y: discretized gaussian mass under mu, sigma evaluated at y_in
            p_y1 = discretized_gaussian_pmf(y1, mu1, sigma1)  
            p_y2 = discretized_gaussian_pmf(y2, mu2, sigma2)  
        if self.distribution == 'Mixture of Gaussians':
            p_y1 = discretized_mixture_pmf(y1, weights1, mus1, sigmas1)
            p_y2 = discretized_mixture_pmf(y2, weights2, mus2, sigmas2)
        logp_y1 = torch.log(p_y1) # [B, M, Hy, Wy], probabilities (training: y_tilde)
        logp_y2 = torch.log(p_y2) # [B, M, Hy, Wy], probabilities (training: y_tilde)
           

        # synthesis / reconstruction uses relaxed or quantized y
        x_hat = self.decoder(y_in)

        # Latent Space Transform
        F_tilde = self.LST(y1)

        out = {
            'x_hat': x_hat,
            'y': y,
            'y_in': y_in,          # y_tilde (train) or y_q (eval)
            'y1': y1,
            'y2': y2,
            'z': z,
            'z_in': z_in,          # z_tilde (train) or z_q (eval)
            'p_z': p_z,
            'logp_z': logp_z,
            'p_y1': p_y1,
            'logp_y1': logp_y1,
            'p_y2': p_y2,
            'logp_y2': logp_y2,
            'F_tilde' : F_tilde,
            'training': training,
        }

        if self.distribution == 'Mean-Scale Gaussian':
            out.update({'mu1': mu1, 'sigma1': sigma1,
                        'mu2': mu2, 'sigma2': sigma2})
        if self.distribution == 'Mixture of Gaussians':
            out.update({'weights1': weights1, 'mus1': mus1, 'sigmas1': sigmas1,
                        'weights2': weights2, 'mus2': mus2, 'sigmas2': sigmas2})
        
        return out

