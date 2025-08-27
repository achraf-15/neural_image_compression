import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

from utils import gaussian_cdf


class EntropyModel(nn.Module):
    """
    Base class for entropy models. Defines the core interface:
    - _likelihood: compute per-element probability mass
    - forward: wrapper returning likelihoods
    - likelihood_bound: lower bound for numerical stability
    """
    def __init__(self, likelihood_lower_bound: float = 1e-9):
        super().__init__()
        self.likelihood_lower_bound = likelihood_lower_bound

    def _likelihood(self, inputs: Tensor, **kwargs) -> Tensor:
        """
        Abstract method: should be implemented by subclasses.
        Returns per-element likelihood (same shape as inputs).
        """
        raise NotImplementedError

    def forward(self, inputs: Tensor, **kwargs) -> Tensor:
        """Compute likelihood clamped by likelihood_bound."""
        return self._likelihood(inputs, **kwargs).clamp_min(self.likelihood_lower_bound)

    def channel_cdf(self, ch: int, x: torch.Tensor) -> torch.Tensor: 
        """Learned CDF for one channel at points x."""
        raise NotImplementedError

    def channel_pmf(self, ch: int, x: torch.Tensor) -> torch.Tensor: 
        """
        Learned discrete PMF for integer bins centered at x (can be real-valued too),
        computed as CDF(x+0.5) - CDF(x-0.5).
        """
        raise NotImplementedError    

    @property
    def likelihood_bound(self) -> float:
        return self.likelihood_lower_bound


class FactorizedEntropyBottleneck(EntropyModel): # Similar to tensorflow TFC implementation
    """
    Entropy bottleneck layer, introduced by J. Ballé, D. Minnen, S. Singh,
    S. J. Hwang, N. Johnston, in `"Variational image compression with a scale
    hyperprior" <https://arxiv.org/abs/1802.01436>`
    """
    def __init__(self, channels: int, init_scale: float = 10.0, hidden_dims: Tuple[int, ...] = (3, 3, 3), likelihood_lower_bound: float = 1e-9):
        super().__init__(likelihood_lower_bound)
        self.channels = int(channels)
        self.init_scale = float(init_scale)
        self.filters = tuple(int(f) for f in hidden_dims)
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

    def _logits_cumulative(self, inputs: Tensor): 
        """
        inputs expected shape: (C, 1, N) where N is flattened batch*spatial.
        returns logits of same shape.
        """
        logits = inputs
        # iterate through layers applying: logits = mat @ logits + bias; if factor, add factor * tanh(logits)
        for i in range(len(self.matrices)):
            matrix_raw = self.matrices[i]
            matrix = F.softplus(matrix_raw)  # ensure positivity similar to TF pattern

            # matrix shape: (C, out, in); logits shape: (C, in, N)
            # perform batch (channel-wise) matmul: for each channel
            logits = torch.matmul(matrix, logits)
            bias = self.biases[i]

            logits = logits + bias  # (C, out, N)
            # if factor exists, apply nonlinearity
            if i < len(self.factors):
                factor = self.factors[i]
                factor_t = torch.tanh(factor)
                logits = logits + factor_t * torch.tanh(logits)

        return logits

    def _likelihood(self, inputs: Tensor):
        """
        inputs: real tensor (B, C, ...) (after noise or dequant)
        returns likelihood per element same shape
        """
        # convert to (C, 1, batch) 
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
        lower = self._logits_cumulative(flat - half)
        upper = self._logits_cumulative(flat + half)

        # sign = -sign(lower + upper); stop gradient through sign
        s = -torch.sign(lower + upper)
        s = s.detach()
        # stable difference: |sigmoid(s * upper) - sigmoid(s * lower)|
        upper_s = torch.sigmoid(s * upper)
        lower_s = torch.sigmoid(s * lower)
        pmf = torch.abs(upper_s - lower_s)  # (C, 1, N)

        # reshape back to original (B, C, *spatial)
        pmf = pmf.view(C, *x.shape[1:])  # (C, B, ...)
        # permute back: original perm was [1,0,...] so invert:
        inv_perm = [1, 0] + list(range(2, inputs.dim()))
        # we want (B, C, ...)
        pmf = pmf.permute(1, 0, *range(2, pmf.dim()))
        return pmf

    @torch.no_grad()
    def channel_logits_cumulative(self, ch: int, x: torch.Tensor) -> torch.Tensor:
        """
        Compute logits of the learned CDF for one channel over x.
        x: 1D tensor of shape [N] on the same device/dtype as the module.
        Returns: logits of shape [N]
        """
        # Slice parameters for the selected channel (faster and simpler than building (C,1,N))
        logits = x.view(1, 1, -1)  # shape (1, 1, N)
        for i in range(len(self.matrices)):
            M = F.softplus(self.matrices[i][ch:ch+1, :, :])  # (1, out, in)
            b = self.biases[i][ch:ch+1, :, :]                                  # (1, out, 1)
            logits = torch.matmul(M, logits) + b                               # (1, out, N)
            if i < len(self.factors):
                f = torch.tanh(self.factors[i][ch:ch+1, :, :])                 # (1, out, 1)
                logits = logits + f * torch.tanh(logits)
        return logits.view(-1)  # (N,)
    
    @torch.no_grad()
    def channel_cdf(self, ch: int, x: torch.Tensor) -> torch.Tensor:
        """Learned CDF for one channel at points x."""
        return torch.sigmoid(self.channel_logits_cumulative(ch, x))
    
    @torch.no_grad()
    def channel_pmf(self, ch: int, x: torch.Tensor) -> torch.Tensor:
        """
        Learned discrete PMF for integer bins centered at x (can be real-valued too),
        computed as CDF(x+0.5) - CDF(x-0.5).
        """
        Lp = self.channel_logits_cumulative(ch, x + 0.5)
        Lm = self.channel_logits_cumulative(ch, x - 0.5)
        return (torch.sigmoid(Lp) - torch.sigmoid(Lm)).clamp_min(1e-12)



class GaussianConditional(EntropyModel):
    def __init__(self, likelihood_lower_bound=1e-9):
        super().__init__(likelihood_lower_bound)

    def discretized_gaussian_pmf(self, x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor):
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
        mass = cdf_upper - cdf_lower
        return mass
    
    def _likelihood(self, x: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
        return self.discretized_gaussian_pmf(x, mu, sigma)
   

class GaussianMixtureConditional(GaussianConditional):
    def __init__(self, likelihood_lower_bound=1e-9):
        super().__init__(likelihood_lower_bound)

    def discretized_mixture_pmf(self, x: torch.Tensor, weights: torch.Tensor, mus: torch.Tensor, sigmas: torch.Tensor):
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
        pmf_per_gauss = self.discretized_gaussian_pmf(x_exp, mus, sigmas)
        
        # Weighted sum across mixture components K
        pmf_mixture = torch.sum(weights * pmf_per_gauss, dim=1)  # → [B, M, H, W]
        return pmf_mixture
    
    def _likelihood(self, x: torch.Tensor, weights: torch.Tensor, mus: torch.Tensor, sigmas: torch.Tensor) -> Tensor:
        return self.discretized_mixture_pmf(x, weights, mus, sigmas)

      