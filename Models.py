import torch
import torch.nn as nn

from Components import Encoder, Decoder, HyperEncoder, HyperDecoder, LatentSpaceTransform
from ContextModels import ContextModel
from EntropyModels import FactorizedEntropyBottleneck, GaussianConditional, GaussianMixtureConditional
from ParametersModels import EntropyParameters


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
        self.conditional = GaussianConditional() if K == 1 else GaussianMixtureConditional()
        
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
        

    def forward(self, x: torch.Tensor, training: bool = True):

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
            params = {"mu": mu, "sigma": sigma}
        if self.distribution == 'Mixture of Gaussians':
            weights, mus, sigmas = self.entropy_parameters(combined) # each [B, K, M, H', W']
            params = {"weights": weights, "mus": mus, "sigmas": sigmas}
        
        # likelihoods:
        p_z = self.factorized_entropy_model(z_in) # [B, M, Hz, Wz], probabilities (training: z_tilde)
        logp_z = torch.log(p_z)
        
        p_y = self.conditional(y_in, **params)
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
        out.update(params)
        
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
        self.conditional = GaussianConditional() if K == 1 else GaussianMixtureConditional()
        
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
            params1 = {"mu1": mu1, "sigma1": sigma1}
            params2 = {"mu2": mu2, "sigma2": sigma2}
        if self.distribution == 'Mixture of Gaussians':
            weights1, mus1, sigmas1 = self.entropy_parameters_1(combined1) # each [B, K, M, H', W']
            weights2, mus2, sigmas2 = self.entropy_parameters_2(combined2) # each [B, K, M, H', W']
            params1 = {"weights1": weights1, "mus1": mus1, "sigmas1": sigmas1}
            params1 = {"weights2": weights2, "mus2": mus2, "sigmas2": sigmas2}
        
        # likelihoods:
        p_z = self.factorized_entropy_model(z_in, debug) # [B, M, Hz, Wz], probabilities (training: z_tilde)
        logp_z = torch.log(p_z)
        
        p_y1 = self.conditional(y1, **params1)
        p_y2 = self.conditional(y2, **params2)

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
        out.update(params1)
        out.update(params2)
        
        return out

