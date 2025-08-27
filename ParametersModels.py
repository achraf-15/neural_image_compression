import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


class EntropyParameters(nn.Module):
    def __init__(self, latent_channels=192, hyper_latent_channels=192, K=1):
        super().__init__()
        
        if not isinstance(K, int) or K < 1:
            raise ValueError(f"K must be int >= 1, got {K}")

        self.K = K
        self.distribution = 'Mean-Scale Gaussian' if K == 1 else 'Mixture of Gaussians'
        self.latent_channels = latent_channels
        self.hyper_latent_channels = hyper_latent_channels

        if self.distribution == 'Mean-Scale Gaussian':
            self.net = nn.Sequential(
                nn.Conv2d(2 * self.latent_channels + 2 * self.hyper_latent_channels, 640, kernel_size=1),
                nn.LeakyReLU(),
                nn.Conv2d(640, 640, kernel_size=1),
                nn.LeakyReLU(),
                nn.Conv2d(640, 2 * self.latent_channels, kernel_size=1)  # outputs [mu|sigma] stacked
            )
        if self.distribution == 'Mixture of Gaussians':
            self.net = nn.Sequential(
                nn.Conv2d(2 * self.latent_channels + 2 * self.hyper_latent_channels, 640, kernel_size=1),
                nn.LeakyReLU(),
                nn.Conv2d(640, 640, kernel_size=1),
                nn.LeakyReLU(),
                nn.Conv2d(640, 3 * self.K * self.latent_channels, kernel_size=1)  # outputs [w_1..w_K|mu_1..mu_K|sigma_1..sigma_K] stacked
            )

    def forward(self, combined_feat: Tensor) -> Tuple[Tensor, Tensor]:
        """
        combined_feat: concat along channel dim of hyperdecoder features and context features; shape [B, phi|psi, H, W]
        returns (mu, scale) each shape [B, M, H, W], scale enforced positive via softplus, if distrubtion is Mean-Scale Gaussian
        returns (weights{1:K}, mus{1:K}, scales{1:K}), each shape [B, M, H, W], scale enforced positive via softplus, if distrubtion is Mixture of Gaussians
        """
        out = self.net(combined_feat)
        
        if self.distribution == 'Mean-Scale Gaussian': 
            mu, sigma = out.chunk(2, dim=1)
            sigma = F.softplus(sigma) + 1e-6 # use softplus/clip sigma to ensure positive scale, and stay away from 0
            return mu, sigma
            
        elif self.distribution == 'Mixture of Gaussians': 
            # Split into weights, mus, sigmas
            weights, mus, sigmas = torch.chunk(out, 3, dim=1)
            
            # Reshape from [B, 3*K*M, H, W] → [B, K, M, H, W]
            weights = weights.view(weights.size(0), self.K, self.latent_channels, *weights.shape[2:])
            mus     = mus.view(mus.size(0), self.K, self.latent_channels, *mus.shape[2:])
            sigmas  = sigmas.view(sigmas.size(0), self.K, self.latent_channels, *sigmas.shape[2:])
            
            # Softmax over K for mixture weights
            weights = F.softmax(weights, dim=1)
            # Ensure sigma > 0
            sigmas = F.softplus(sigmas) + 1e-6
            
            return weights, mus, sigmas