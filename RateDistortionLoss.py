import math
import torch
import torch.nn as nn

def rd_loss(model_out: dict, x: torch.Tensor, lambda_rd: float):
    """
    Compute R (bits per pixel), D (MSE), and L = R + lambda * D.

    model_out: output dict from JointAutoregressiveHierarchical.forward(...)
    x: ground-truth image tensor [B, C, H, W]
    lambda_rd: Lagrange multiplier

    Returns: dict with 'loss', 'bpp_y', 'bpp_z', 'bpp_total', 'mse'
    """
    B = x.size(0)
    eps = 1e-8
    # get logp in nats (we used natural log earlier)
    logp_y = model_out['logp_y']   # [B, M, Hy, Wy]
    logp_z = model_out['logp_z']   # [B, M, Hz, Wz]
    # sum over channels and spatial dims -> nats per image
    nats_y_per_image = -torch.sum(logp_y, dim=(1,2,3))  # shape [B]
    nats_z_per_image = -torch.sum(logp_z, dim=(1,2,3))  # shape [B]
    # convert to bits
    bits_y_per_image = nats_y_per_image / math.log(2.0)
    bits_z_per_image = nats_z_per_image / math.log(2.0)
    # pixels in original image (assuming decoder output has same H,W as x)
    num_pixels = x.size(2) * x.size(3)
    bpp_y = (bits_y_per_image / num_pixels).mean() # bits_y per pixel per image
    bpp_z = (bits_z_per_image / num_pixels).mean() # bits_z per pixel per image
    bpp_total = (bpp_y + bpp_z) # bits_y per pixel per image


    # distortion: MSE between reconstruction and original (use per-image then mean)
    mse_per_image = torch.mean((model_out['x_hat'] - x) ** 2, dim=(1, 2, 3))
    mse = mse_per_image.mean()
    
    # convert to PSNR for evaluation
    psnr = -10 * torch.log10(mse + eps)
    psnr_per_image = -10 * torch.log10(mse_per_image + eps)

    # Lagrangian loss
    loss = bpp_total + lambda_rd * mse

    return {
        'loss': loss,
        'bpp_y': bpp_y.item(),
        'bpp_z': bpp_z.item(),
        'bpp_total': bpp_total.item(),
        'mse': mse.item(),
        'psnr': psnr.item(),
        'mse_per_image': mse_per_image.detach(),
        'psnr_per_image': psnr_per_image.detach(),
        'bits_y': bits_y_per_image.mean().item(),
        'bits_z': bits_z_per_image.mean().item(),
        'bits_total': (bits_y_per_image + bits_z_per_image).mean().item(),
        
    }