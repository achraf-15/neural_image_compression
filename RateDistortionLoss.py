import math
import torch
import torch.nn as nn

def rd_loss(model_out: dict, x: torch.Tensor, lambda_rd: float):

    B = x.size(0)
    eps = 1e-8
    # get logp in nats (we used natural log earlier)
    logp_y = model_out['logp_y']   # [B, M, Hy, Wy]
    logp_z = model_out['logp_z']   # [B, M, Hz, Wz]
    # sum over channels and spatial dims 
    nats_y_per_image = -torch.sum(logp_y, dim=(1,2,3))  # shape [B]
    nats_z_per_image = -torch.sum(logp_z, dim=(1,2,3))  # shape [B]
    # convert to bits
    bits_y_per_image = nats_y_per_image / math.log(2.0)
    bits_z_per_image = nats_z_per_image / math.log(2.0)
    # pixels in original image
    num_pixels = x.size(2) * x.size(3)
    bpp_y = (bits_y_per_image / num_pixels).mean() # bits_y per pixel per image
    bpp_z = (bits_z_per_image / num_pixels).mean() # bits_z per pixel per image
    bpp_total = (bpp_y + bpp_z) # bits_y per pixel per image


    # distortion: MSE between reconstruction and original 
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


def vision_rd_loss(model_out: dict, x: torch.Tensor, lambda_rd: float, gamma: float, frozen_activation: nn.Module = None, V: nn.Module = None):
    
    B = x.size(0)
    eps = 1e-8
    # get logp in nats (we used natural log earlier)
    logp_y1 = model_out['logp_y1']   # [B, M, Hy, Wy]
    logp_y2 = model_out['logp_y2']   # [B, M, Hy, Wy]
    logp_z = model_out['logp_z']   # [B, M, Hz, Wz]
    # sum over channels and spatial dims 
    nats_y1_per_image = -torch.sum(logp_y1, dim=(1,2,3))  # shape [B]
    nats_y2_per_image = -torch.sum(logp_y2, dim=(1,2,3))  # shape [B]
    nats_z_per_image = -torch.sum(logp_z, dim=(1,2,3))  # shape [B]
    # convert to bits
    bits_y1_per_image = nats_y1_per_image / math.log(2.0)
    bits_y2_per_image = nats_y2_per_image / math.log(2.0)
    bits_y_per_image = bits_y1_per_image + bits_y2_per_image
    bits_z_per_image = nats_z_per_image / math.log(2.0)
    # pixels in original image 
    num_pixels = x.size(2) * x.size(3)
    bpp_y1 = (bits_y1_per_image / num_pixels).mean() # bits_y per pixel per image
    bpp_y2 = (bits_y2_per_image / num_pixels).mean() # bits_y per pixel per image
    bpp_y = bpp_y1 + bpp_y2 # bits_y per pixel per image
    bpp_z = (bits_z_per_image / num_pixels).mean() # bits_z per pixel per image
    bpp_total = (bpp_y1 + bpp_y2 + bpp_z) # bits_y per pixel per image


    # distortion: MSE between reconstruction and original 
    reconstruction_mse_per_image = torch.mean((model_out['x_hat'] - x) ** 2, dim=(1, 2, 3))
    mse_per_image = reconstruction_mse_per_image
    reconstruction_mse = reconstruction_mse_per_image.mean()
    mse = reconstruction_mse
    
    # convert to PSNR for evaluation
    psnr = -10 * torch.log10(reconstruction_mse + eps)
    psnr_per_image = -10 * torch.log10(reconstruction_mse_per_image + eps)

    vision_mse = 0.0
    if frozen_activation is not None and V is not None:
        F_activated = frozen_activation(model_out['F_tilde'])
        F = V(model_out['x_hat'])
        vision_mse_per_image = torch.mean((F_activated - F) ** 2, dim=(1,2,3))
        vision_mse = vision_mse_per_image.mean()
        mse_per_image = reconstruction_mse_per_image + gamma * vision_mse_per_image  
        mse = reconstruction_mse + gamma * vision_mse

    # Lagrangian loss
    loss = bpp_total + lambda_rd * mse

    return {
        'loss': loss,
        'bpp_y1': bpp_y1.item(),
        'bpp_y2': bpp_y2.item(),
        'bpp_y': bpp_y.item(),
        'bpp_z': bpp_z.item(),
        'bpp_total': bpp_total.item(),
        'mse': mse.item(),
        'reconstruction_mse': reconstruction_mse.item(),
        'psnr': psnr.item(),
        'vision_mse' : vision_mse.item() if isinstance(V, nn.Module) else vision_mse,
        'mse_per_image': mse_per_image.detach(),
        'reconstruction_mse_per_image': reconstruction_mse_per_image.detach(),
        'psnr_per_image': psnr_per_image.detach(),
        'vision_mse_per_image': vision_mse_per_image.detach() if isinstance(V, nn.Module) else 0.0,
        'bits_y1': bits_y1_per_image.mean().item(),
        'bits_y2': bits_y2_per_image.mean().item(),
        'bits_y': bits_y_per_image.mean().item(),
        'bits_z': bits_z_per_image.mean().item(),
        'bits_total': (bits_y_per_image + bits_z_per_image).mean().item(),
        
    }