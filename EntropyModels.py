import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

class FactorizedEntropyBottleneck(nn.Module):
    def __init__(self, channels: int, init_scale=10.0, hidden_dims: Tuple[int, ...] = (3, 3, 3)):
        super().__init__()
        self.channels = int(channels)
        self.init_scale = float(init_scale)
        self.filters = tuple(int(f) for f in hidden_dims)
        #K = 1 + len(self.r)         # total layers (last is sigmoid)
        self.dtype = torch.float32

        filters_full = (1,) + self.filters + (1,)
        scale = self.init_scale ** (1.0 / (len(self.filters) + 1))

        # matrices: list length len(filters)+1, shapes: (C, out, in)
        self.matrices = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.factors = nn.ParameterList()

        for i in range(len(self.filters) + 1):
            out = filters_full[i+1]
            inp = filters_full[i]
            # TFC used special init: consant init for all values in matrices_params = np.log(np.expm1(1 / scale / filters[i + 1]))
            init_val = math.log(math.expm1(1.0 / scale / out))
            p = nn.Parameter(torch.full((self.channels, out, inp), init_val, dtype=self.dtype))
            # store raw param before softplus; forward will apply softplus
            self.matrices.append(p)

            b = nn.Parameter(torch.empty((self.channels, out, 1), dtype=self.dtype))
            nn.init.uniform_(b, -0.5, 0.5)
            self.biases.append(b)

            if i < len(self.filters):
                f = nn.Parameter(torch.zeros((self.channels, out, 1), dtype=self.dtype))
                # TF applies tanh to factor; we'll apply tanh in forward
                self.factors.append(f)

    def _logits_cumulative(self, inputs: Tensor, stop_gradient: bool, debug=False):
        """
        inputs expected shape: (C, 1, N) where N is flattened batch*spatial.
        returns logits of same shape.
        """
        logits = inputs
        # iterate through layers applying: logits = mat @ logits + bias; if factor, add factor * tanh(logits)
        for i in range(len(self.matrices)):
            matrix_raw = self.matrices[i]
            matrix = F.softplus(matrix_raw)  # ensure positivity similar to TF pattern
            if stop_gradient:
                matrix = matrix.detach()
            # matrix shape: (C, out, in); logits shape: (C, in, N)
            # perform batch (channel-wise) matmul: for each channel
            logits = torch.matmul(matrix, logits)
            bias = self.biases[i]
            if stop_gradient:
                bias = bias.detach()
            logits = logits + bias  # (C, out, N)
            # if factor exists, apply nonlinearity
            if i < len(self.factors):
                factor = self.factors[i]
                if stop_gradient:
                    factor = factor.detach()
                factor_t = torch.tanh(factor)
                logits = logits + factor_t * torch.tanh(logits)

        if debug:
            for i, M in enumerate(self.matrices):
                M_sp = F.softplus(M)
                print(f"matrix {i} softplus: min {M_sp.min().item():.3e}, max {M_sp.max().item():.3e}")

        return logits

    def _likelihood(self, inputs: Tensor, debug=False):
        """
        inputs: real tensor (B, C, ...) (after noise or dequant)
        returns likelihood per element same shape
        """
        # convert to (C, 1, batch) collapsed form similar to TF
        # move channel to front
        shape = inputs.shape
        if inputs.dim() < 2:
            raise ValueError("inputs must be at least 2D with channel axis")
        B = shape[0]
        # Move C to front, flatten rest (including batch) to last dim
        # We'll preserve channel-first assumption: inputs (B, C, *spatial)
        # Create flattened view: (C, 1, B * prod(spatial))
        C = self.channels
        # Permute to (C, B, ...spatial)
        perm = [1, 0] + list(range(2, inputs.dim()))
        x = inputs.permute(perm).contiguous()
        flat = x.view(C, 1, -1)  # shape (C, 1, N)

        half = 0.5
        lower = self._logits_cumulative(flat - half, stop_gradient=False)
        upper = self._logits_cumulative(flat + half, stop_gradient=False, debug=debug)

        # sign trick: choose sign so subtraction occurs in left tail
        # sign = -sign(lower + upper); stop gradient through sign
        s = -torch.sign(lower + upper)
        s = s.detach()
        # stable difference: |sigmoid(s * upper) - sigmoid(s * lower)|
        upper_s = torch.sigmoid(s * upper)
        lower_s = torch.sigmoid(s * lower)
        pmf = torch.abs(upper_s - lower_s)  # (C, 1, N)

        # Debuging: check numeric ranges/gradients
        if debug:
            with torch.no_grad():
                print("pmf min, max:", pmf.min().item(), pmf.max().item())
                print("lower logits min/max:", lower.min().item(), lower.max().item())
                print("upper logits min/max:", upper.min().item(), upper.max().item())
 

        
        # reshape back to original (B, C, *spatial)
        pmf = pmf.view(C, *x.shape[1:])  # (C, B, ...)
        # permute back: original perm was [1,0,...] so invert:
        inv_perm = [1, 0] + list(range(2, inputs.dim()))
        # we want (B, C, ...)
        pmf = pmf.permute(1, 0, *range(2, pmf.dim()))
        return pmf

    def forward(self, x: torch.Tensor, debug=False, eps: float=1e-12) -> torch.Tensor:
        """
        x: [B, C, H, W] -> returns cdf(x) in (0,1), same shape.
        """
        return self._likelihood(x, debug).clamp_min(eps)


class EntropyParameters(nn.Module):
    def __init__(self, latent_channels=192, K=1):
        super().__init__()
        
        if not isinstance(K, int) or K < 1:
            raise ValueError(f"K must be int >= 1, got {K}")

        self.K = K
        self.distribution = 'Mean-Scale Gaussian' if K == 1 else 'Mixture of Gaussians'
        self.latent_channels = latent_channels

        if self.distribution == 'Mean-Scale Gaussian':
            self.net = nn.Sequential(
                nn.Conv2d(4 * self.latent_channels, 640, kernel_size=1),
                nn.LeakyReLU(),
                nn.Conv2d(640, 640, kernel_size=1),
                nn.LeakyReLU(),
                nn.Conv2d(640, 2 * self.latent_channels, kernel_size=1)  # outputs [mu|sigma] stacked
            )
        if self.distribution == 'Mixture of Gaussians':
            self.net = nn.Sequential(
                nn.Conv2d(4 * self.latent_channels, 640, kernel_size=1),
                nn.LeakyReLU(),
                nn.Conv2d(640, 640, kernel_size=1),
                nn.LeakyReLU(),
                nn.Conv2d(640, 3 * self.K * self.latent_channels, kernel_size=1)  # outputs [w_1..w_K|mu_1..mu_K|sigma_1..sigma_K] stacked
            )

    def forward(self, combined_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
            

# ---------------------
# Discretized Gaussian mass (CDF difference)
# ---------------------
def gaussian_cdf(x: torch.Tensor):
    """Standard normal CDF via erf"""
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def discretized_gaussian_pmf(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, eps: float=1e-12):
    """
    Compute P(bin) using Gaussian CDF differences.
    x: relaxed latent value (y_tilde) or integer centers at eval time (y_hat)
    mu, sigma: broadcastable to x shape
    returns: probability of the discrete bin containing x
    """
    upper = (x + 0.5 - mu) / sigma
    lower = (x - 0.5 - mu) / sigma
    cdf_upper = gaussian_cdf(upper)
    cdf_lower = gaussian_cdf(lower)
    mass = (cdf_upper - cdf_lower).clamp_min(1e-12)
    return mass

def discretized_mixture_pmf(x: torch.Tensor, weights: torch.Tensor, mus: torch.Tensor, sigmas: torch.Tensor, eps: float=1e-12):
    """
    Mixture of Gaussians PMF.
    x:      [B, M, H, W]
    weights: [B, K, M, H, W]
    mus:     [B, K, M, H, W]
    sigmas:  [B, K, M, H, W]
    """
    # Expand x for broadcasting: [B, 1, M, H, W]
    x_exp = x.unsqueeze(1)
    
    # PMF per Gaussian: [B, K, M, H, W]
    pmf_per_gauss = discretized_gaussian_pmf(x_exp, mus, sigmas, eps)
    
    # Weighted sum across mixture components K
    pmf_mixture = torch.sum(weights * pmf_per_gauss, dim=1)  # → [B, M, H, W]
    return pmf_mixture.clamp_min(eps)

# ---------------------
# Discretized sigmoid mass (CDF difference)
# ---------------------
def discretized_sigmoid_pmf(x: torch.Tensor, eps: float=1e-12):
    """
    Compute P(bin) using a sigmoid CDF differences (We use the sigmoid function as replacement of the CDF: CDF(x) = sigmoid(x) in [0,1])
    x: relaxed latent value (y_tilde) or integer centers at eval time (y_hat)
    returns: probability of the discrete bin containing x
    """
    upper = (x + 0.5) 
    lower = (x - 0.5) 
    cdf_upper = torch.sigmoid(upper)
    cdf_lower = torch.sigmoid(lower)
    mass = (cdf_upper - cdf_lower).clamp_min(1e-12)
    return mass