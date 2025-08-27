import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import os
from pytorch_msssim import ms_ssim

def normalize_map(x, method="minmax"):
    x = x.astype(np.float32)
    if method == "minmax":
        return (x - x.min()) / (x.max() - x.min() + 1e-12)
    elif method == "std":
        return (x - x.mean()) / (x.std() + 1e-12)
    return x

class CompressionEvaluator:
    def __init__(self, model, dataloader, device, lambda_val, save_dir="./eval_results"):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.lambda_val = lambda_val
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

    @staticmethod
    def rgb_to_luma(x):
        # x: [B, 3, H, W], values in [0,1]
        R, G, B = x[:,0], x[:,1], x[:,2]
        return 0.299*R + 0.587*G + 0.114*B

    def compute_metrics(self, orig, recon):
        # orig, recon: [1,3,H,W], values in [0,1]
        mse_rgb = torch.mean((orig - recon) ** 2).item()
        mse_rgb_255 = mse_rgb * (255**2)

        psnr_rgb = 10 * np.log10(1.0 / mse_rgb) if mse_rgb > 0 else float('inf')
        msssim_rgb = ms_ssim(recon, orig, data_range=1.0, size_average=True).item()

        # Luma metrics
        Y_orig = self.rgb_to_luma(orig).unsqueeze(1)
        Y_recon = self.rgb_to_luma(recon).unsqueeze(1)
        mse_y = torch.mean((Y_orig - Y_recon) ** 2).item()
        psnr_y = 10 * np.log10(1.0 / mse_y) if mse_y > 0 else float('inf')
        msssim_y = ms_ssim(Y_recon, Y_orig, data_range=1.0, size_average=True).item()

        return {
            "MSE(255)": mse_rgb_255,
            "PSNR(RGB)": psnr_rgb,
            "MS-SSIM(RGB)": msssim_rgb,
            "PSNR(Y)": psnr_y,
            "MS-SSIM(Y)": msssim_y
        }

    def evaluate(self, rd_loss_fn):
        self.model.eval()
        total_metrics = []
        bbp_values, bpp_y_values, bpp_z_values = [], [], []
        imgs_list, recon_list = [], []

        with torch.no_grad():
            for imgs in self.dataloader:
                imgs = imgs.to(self.device)
                out = self.model(imgs, training=False)
                results = rd_loss_fn(out, imgs, self.lambda_val)

                # Rate stats
                bbp_values.append(results["bpp_total"])
                bpp_y_values.append(results["bpp_y"])
                bpp_z_values.append(results["bpp_z"])

                # Distortion stats
                metrics = self.compute_metrics(imgs, out["x_hat"].clamp(0, 1))
                total_metrics.append(metrics)

                imgs_list.append(imgs[0].cpu())
                recon_list.append(out["x_hat"][0].cpu().clamp(0, 1))

        # Aggregate
        avg_metrics = {k: np.mean([m[k] for m in total_metrics]) for k in total_metrics[0]}
        bpp_total_values = np.array(bpp_y_values)
        bpp_y_values = np.array(bpp_y_values)
        bpp_z_values = np.array(bpp_z_values)
        avg_metrics['BPP'] = bpp_total_values.mean()
        avg_metrics['BPP(y)'] = bpp_y_values.mean()
        avg_metrics['BPP(z)'] = bpp_z_values.mean()

        print("\n--- Evaluation Results ---")
        for k, v in avg_metrics.items():
            print(f"{k}: {v:.6f}")

        return avg_metrics, imgs_list, recon_list

    def plot_samples(self, imgs_list, recon_list, rd_loss_fn, n=3):
        indices = random.sample(range(len(imgs_list)), n)
        for idx in indices:
            orig = imgs_list[idx].permute(1, 2, 0).numpy()
            recon = recon_list[idx].permute(1, 2, 0).numpy()
                
            with torch.no_grad():
                out = self.model(imgs_list[idx].unsqueeze(0).to(self.device), training=False)
                results = rd_loss_fn(out, imgs_list[idx].unsqueeze(0).to(self.device), self.lambda_val)
                bbp = results["bpp_total"]
                byte = math.ceil(results["bits_total"]/8)

            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(orig)
            plt.title("Original")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(recon)
            plt.title(f"Reconstructed: {byte} bytes ({bbp:.4f} bit/px)")
            plt.axis("off")
            plt.show()

    def plot_high_entropy_channel(self, imgs_list):
        idx = random.randint(0, len(imgs_list)-1)
        img = imgs_list[idx].unsqueeze(0).to(self.device)
        img_np = imgs_list[idx].permute(1, 2, 0).cpu().numpy()
        
        with torch.no_grad():
            out = self.model(img, training=False)
        
        latents = out["y"][0]  # [C,H,W] # y or y_in ??
        entropies = out["logp_y"][0]  # [C,H,W]
        mean_entropy_per_channel = entropies.view(latents.shape[0], -1).mean(dim=1)

        hyper_latents = out["z"][0 ]  # [C,H,W] # z or z_in ??
        hyper_entropies = out["logp_z"][0] 
        mean_entropy_per_z_channel = hyper_entropies.view(hyper_latents.shape[0], -1).mean(dim=1)
    
        
        # Pick highest entropy channel ( since entropy = -logp, we use argmin not argmax ) 
        high_c = torch.argmin(mean_entropy_per_channel).item()
        high_cz = torch.argmin(mean_entropy_per_z_channel).item()
        
        # Case 1: Mean-Scale Gaussian (K=1)
        if "mu" in out and "sigma" in out:
            mean = out["mu"][0, high_c]
            scale = out["sigma"][0, high_c]
            latent = latents[high_c]
            hyper_latent = hyper_latents[high_cz]
            norm_latent = (latent - mean) / (scale + 1e-12)  
            latent_entropy = -entropies[high_c] / math.log(2.0) 
            hyper_entropy = -hyper_entropies[high_cz] / math.log(2.0) 
            
            maps = [
                ("Original", img_np),
                ("Latent", normalize_map(latent.cpu().numpy())),
                ("Predicted Mean", normalize_map(mean.cpu().numpy())),
                ("Predicted Scale", normalize_map(scale.cpu().numpy())),
                ("Normalized Latent", normalize_map(norm_latent.cpu().numpy())),
                ("Latent Entropy", normalize_map(latent_entropy.cpu().numpy())),
                ("Hyper Latent", hyper_latent.cpu().numpy()),
                ("Hyper Entropy", hyper_entropy.cpu().numpy())
            ]
            
            fig, axes = plt.subplots(1, len(maps), figsize=(3*len(maps), 3))
            for ax, (title, data) in zip(axes, maps):
                if title == "Original":
                    ax.imshow(data)
                else:    
                    if title == "Latent Entropy" or title == "Hyper Entropy":
                        im = ax.imshow(data, cmap="viridis", vmin=0)
                    else:
                        im = ax.imshow(data, cmap="viridis")
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title(title)
                ax.axis("off")
            plt.show()
        
        # Case 2: Mixture of Gaussians (K>1)
        elif "weights" in out and "mus" in out and "sigmas" in out:
            weights = out["weights"][0, :, high_c]     # [K,H,W]
            mus = out["mus"][0, :, high_c]             # [K,H,W]
            sigmas = out["sigmas"][0, :, high_c]       # [K,H,W]
            latent = latents[high_c]                # [H,W]
            hyper_latent = hyper_latents[high_cz]     # [H,W]
            norm_latent = (latent - mus) / (sigmas + 1e-12) # should be [K,H,W], normalized latents for each component
            latent_entropy = -entropies[high_c] / math.log(2.0)
            #print(latent_entropy.min().item(), latent_entropy.mean().item(), latent_entropy.max().item()) 
            hyper_entropy = -hyper_entropies[high_cz] / math.log(2.0) 
            #print(hyper_entropy.min().item(), hyper_entropy.mean().item(), hyper_entropy.max().item()) 
            K = mus.shape[0]
            
            # One row per component
            fig, axes = plt.subplots(K+1, 6, figsize=(18, 3*(K+1)))
            for k in range(K):
                maps = [
                    (f"Comp {k} Weight", normalize_map(weights[k].cpu().numpy())),
                    (f"Comp {k} Mean", normalize_map(mus[k].cpu().numpy())),
                    (f"Comp {k} Sigma", normalize_map(sigmas[k].cpu().numpy())),
                    (f"Comp {k} Norm Latent", normalize_map(norm_latent[k].cpu().numpy())),
                    ("Latent", normalize_map(latent.cpu().numpy())),
                    ("Hyper Latent", normalize_map(hyper_latent.cpu().numpy()))
                ]
                for ax, (title, data) in zip(axes[k], maps):
                    im = ax.imshow(data, cmap="viridis")
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    ax.set_title(title)
                    ax.axis("off")
            
            # Mixture weighted mean & scale
            mixture_mean = torch.sum(weights * mus, dim=0)  # [H,W]
            mixture_var = torch.sum(weights * (sigmas**2 + mus**2), dim=0) - mixture_mean**2
            mixture_sigma = torch.sqrt(mixture_var.clamp(min=1e-9))
            mixture_norm = (latent - mixture_mean) / mixture_sigma 
            
            maps = [
                ("Original", img_np),
                ("Mixture Mean", normalize_map(mixture_mean.cpu().numpy())),
                ("Mixture Sigma", normalize_map(mixture_sigma.cpu().numpy())),
                ("Mixture Norm", normalize_map(mixture_norm.cpu().numpy())),
                ("Latent Entropy", latent_entropy.cpu().numpy()),
                ("Hyper Entropy", hyper_entropy.cpu().numpy()),   
            ]
            for ax, (title, data) in zip(axes[K], maps):
                if title == "Original":
                    ax.imshow(data)
                else:
                    if title == "Latent Entropy" or title == "Hyper Entropy":
                        im = ax.imshow(data, cmap="viridis", vmin=0)
                    else:
                        im = ax.imshow(data, cmap="viridis")
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title(title)
                ax.axis("off")
            
            plt.tight_layout()
            plt.show()


    def save_results(self, metrics, nb_steps, caption=""):
        path = os.path.join(self.save_dir, f"eval_results_{self.lambda_val}_lambda_"+caption+".txt")
        with open(path, "w") as f:
            f.write(f"Lambda: {self.lambda_val}\n")
            f.write(f"Trained for: {nb_steps} steps\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v:.6f}\n")
        print(f"Results saved to {path}")


class VisionCompressionEvaluator:
    def __init__(self, model, dataloader, device, lambda_val, gamma, save_dir="./eval_results"):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.lambda_val = lambda_val
        self.gamma = gamma
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

    @staticmethod
    def rgb_to_luma(x):
        # x: [B, 3, H, W], values in [0,1]
        R, G, B = x[:,0], x[:,1], x[:,2]
        return 0.299*R + 0.587*G + 0.114*B

    def compute_metrics(self, orig, recon):
        # orig, recon: [1,3,H,W], values in [0,1]
        mse_rgb = torch.mean((orig - recon) ** 2).item()
        mse_rgb_255 = mse_rgb * (255**2)

        psnr_rgb = 10 * np.log10(1.0 / mse_rgb) if mse_rgb > 0 else float('inf')
        msssim_rgb = ms_ssim(recon, orig, data_range=1.0, size_average=True).item()

        # Luma metrics
        Y_orig = self.rgb_to_luma(orig).unsqueeze(1)
        Y_recon = self.rgb_to_luma(recon).unsqueeze(1)
        mse_y = torch.mean((Y_orig - Y_recon) ** 2).item()
        psnr_y = 10 * np.log10(1.0 / mse_y) if mse_y > 0 else float('inf')
        msssim_y = ms_ssim(Y_recon, Y_orig, data_range=1.0, size_average=True).item()

        return {
            "MSE(255)": mse_rgb_255,
            "PSNR(RGB)": psnr_rgb,
            "MS-SSIM(RGB)": msssim_rgb,
            "PSNR(Y)": psnr_y,
            "MS-SSIM(Y)": msssim_y
        }

    def evaluate(self, vision_rd_loss):
        self.model.eval()
        total_metrics = []
        bbp_values, bpp_y_values, bpp_z_values = [], [], []
        bpp_y1_values, bpp_y2_values = [], []
        imgs_list, recon_list = [], []

        with torch.no_grad():
            for imgs in self.dataloader:
                imgs = imgs.to(self.device)
                out = self.model(imgs, training=False)
                results = vision_rd_loss(out, imgs, self.lambda_val, self.gamma)

                # Rate stats
                bbp_values.append(results["bpp_total"])
                bpp_y_values.append(results["bpp_y"])
                bpp_y1_values.append(results["bpp_y1"])
                bpp_y2_values.append(results["bpp_y2"])
                bpp_z_values.append(results["bpp_z"])

                # Distortion stats
                metrics = self.compute_metrics(imgs, out["x_hat"].clamp(0, 1))
                total_metrics.append(metrics)

                imgs_list.append(imgs[0].cpu())
                recon_list.append(out["x_hat"][0].cpu().clamp(0, 1))

        # Aggregate
        avg_metrics = {k: np.mean([m[k] for m in total_metrics]) for k in total_metrics[0]}
        bpp_total_values = np.array(bpp_y_values)
        bpp_y_values = np.array(bpp_y_values)
        bpp_y1_values = np.array(bpp_y1_values)
        bpp_y2_values = np.array(bpp_y2_values)
        bpp_z_values = np.array(bpp_z_values)
        avg_metrics['BPP'] = bpp_total_values.mean()
        avg_metrics['BPP(y)'] = bpp_y_values.mean()
        avg_metrics['BPP(y1)'] = bpp_y1_values.mean()
        avg_metrics['BPP(y2)'] = bpp_y2_values.mean()
        avg_metrics['BPP(z)'] = bpp_z_values.mean()

        print("\n--- Evaluation Results ---")
        for k, v in avg_metrics.items():
            print(f"{k}: {v:.6f}")

        return avg_metrics, imgs_list, recon_list

    def plot_samples(self, imgs_list, recon_list, vision_rd_loss, n=3):
        indices = random.sample(range(len(imgs_list)), n)
        for idx in indices:
            orig = imgs_list[idx].permute(1, 2, 0).numpy()
            recon = recon_list[idx].permute(1, 2, 0).numpy()
                
            with torch.no_grad():
                out = self.model(imgs_list[idx].unsqueeze(0).to(self.device), training=False)
                results = vision_rd_loss(out, imgs_list[idx].unsqueeze(0).to(self.device), self.lambda_val, self.gamma)
                bbp = results["bpp_total"]
                byte = math.ceil(results["bits_total"]/8)

            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(orig)
            plt.title("Original")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(recon)
            plt.title(f"Reconstructed: {byte} bytes ({bbp:.4f} bit/px)")
            plt.axis("off")
            plt.show()

    def plot_high_entropy_channel(self,imgs_list, latent_idx): #latent_idx=1 for base latents, and latent_idx=2 for enhacement latents
        idx = random.randint(0, len(imgs_list)-1)
        img = imgs_list[idx].unsqueeze(0).to(self.device)
        img_np = imgs_list[idx].permute(1, 2, 0).cpu().numpy()
        
        with torch.no_grad():
            out = self.model(img, training=False)
        
        latents = out["y"+str(latent_idx)][0]  # [C,H,W]
        entropy = -out["logp_y"+str(latent_idx)][0] / math.log(2.0)  # [C,H,W]
        mean_entropy_per_channel = entropy.view(latents.shape[0], -1).mean(dim=1)
        
        # Pick highest entropy channel
        high_c = torch.argmax(mean_entropy_per_channel).item()
        
        # Case 1: Mean-Scale Gaussian (K=1)
        if "mu"+str(latent_idx) in out and "sigma"+str(latent_idx) in out:
            mean = out["mu"+str(latent_idx)][0, high_c]
            scale = out["sigma"+str(latent_idx)][0, high_c]
            latent = latents[high_c]
            pred_error = latent - mean
            latent_entropy = entropy[high_c]
            hyper_entropy = out["logp_z"][0, high_c]

            if latent_idx == 1:
                maps = [
                    ("Original", img_np),
                    ("Base Latent", latent.cpu().numpy()),
                    ("Predicted Base Mean", mean.cpu().numpy()),
                    ("Base Prediction Error", pred_error.cpu().numpy()),
                    ("Predicted Base Scale", scale.cpu().numpy()),
                    ("Latent Base Entropy", latent_entropy.cpu().numpy()),
                    ("Hyper Entropy", hyper_entropy.cpu().numpy())
                ]
            if latent_idx == 2:
                maps = [
                    ("Original", img_np),
                    ("Enh. Latent", latent.cpu().numpy()),
                    ("Predicted Enh. Mean", mean.cpu().numpy()),
                    ("Enh. Prediction Error", pred_error.cpu().numpy()),
                    ("Predicted Enh. Scale", scale.cpu().numpy()),
                    ("Latent Enh. Entropy", latent_entropy.cpu().numpy()),
                    ("Hyper Entropy", hyper_entropy.cpu().numpy())
                ]
            
            
            fig, axes = plt.subplots(1, len(maps), figsize=(3*len(maps), 3))
            for ax, (title, data) in zip(axes, maps):
                if title == "Original":
                    ax.imshow(data)
                else:
                    im = ax.imshow(data, cmap="viridis")
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title(title)
                ax.axis("off")
            plt.show()
        
        # Case 2: Mixture of Gaussians (K>1)
        elif "weights"+str(latent_idx) in out and "mus"+str(latent_idx) in out and "sigmas"+str(latent_idx) in out:
            weights = out["weights"+str(latent_idx)][0, :, high_c]     # [K,H,W]
            mus = out["mus"+str(latent_idx)][0, :, high_c]             # [K,H,W]
            sigmas = out["sigmas"+str(latent_idx)][0, :, high_c]       # [K,H,W]
            latent = latents[high_c]                # [H,W]
            latent_entropy = -entropy[high_c] / math.log(2.0)
            hyper_entropy = -out["logp_z"][0, high_c] / math.log(2.0)
            K = mus.shape[0]
            
            # One row per component
            fig, axes = plt.subplots(K+1, 6, figsize=(18, 3*(K+1)))
            for k in range(K):
                pred_error = latent - mus[k]
                if latent_idx == 1:
                    maps = [
                        (f"Base Comp {k} Weight", weights[k].cpu().numpy()),
                        (f"Base Comp {k} Mean", mus[k].cpu().numpy()),
                        (f"Base Comp {k} Pred Error", pred_error.cpu().numpy()),
                        (f"Base Comp {k} Sigma", sigmas[k].cpu().numpy()),
                        ("Latent Base Entropy", latent_entropy.cpu().numpy()),
                        ("Hyper Entropy", hyper_entropy.cpu().numpy())
                    ]
                if latent_idx == 2:
                    maps = [
                        (f"Enh. Comp {k} Weight", weights[k].cpu().numpy()),
                        (f"Enh. Comp {k} Mean", mus[k].cpu().numpy()),
                        (f"Enh. Comp {k} Pred Error", pred_error.cpu().numpy()),
                        (f"Enh. Comp {k} Sigma", sigmas[k].cpu().numpy()),
                        ("Latent Enh. Entropy", latent_entropy.cpu().numpy()),
                        ("Hyper Entropy", hyper_entropy.cpu().numpy())
                    ]
                for ax, (title, data) in zip(axes[k], maps):
                    im = ax.imshow(data, cmap="viridis")
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    ax.set_title(title)
                    ax.axis("off")
            
            # Mixture weighted mean & scale
            mixture_mean = torch.sum(weights * mus, dim=0)  # [H,W]
            mixture_var = torch.sum(weights * (sigmas**2 + mus**2), dim=0) - mixture_mean**2
            mixture_sigma = torch.sqrt(mixture_var.clamp(min=1e-9))
            
            mixture_error = latent - mixture_mean
            if latent_idx == 1:
                maps = [
                    ("Original", img_np),
                    ("Base Mixture Mean", mixture_mean.cpu().numpy()),
                    ("Base Mixture Sigma", mixture_sigma.cpu().numpy()),
                    ("Base Mixture Error", mixture_error.cpu().numpy()),
                    ("Latent Base Entropy", latent_entropy.cpu().numpy()),
                    ("Hyper Entropy", hyper_entropy.cpu().numpy()),   
                ]
            if latent_idx == 2:
                maps = [
                    ("Original", img_np),
                    ("Enh. Mixture Mean", mixture_mean.cpu().numpy()),
                    ("Enh. Mixture Sigma", mixture_sigma.cpu().numpy()),
                    ("Enh. Mixture Error", mixture_error.cpu().numpy()),
                    ("Latent Enh. Entropy", latent_entropy.cpu().numpy()),
                    ("Hyper Entropy", hyper_entropy.cpu().numpy()),   
                ]
            for ax, (title, data) in zip(axes[K], maps):
                if title == "Original":
                    ax.imshow(data)
                else:
                    im = ax.imshow(data, cmap="viridis")
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title(title)
                ax.axis("off")
            
            plt.tight_layout()
            plt.show()


    def save_results(self, metrics, nb_steps, caption=""):
        path = os.path.join(self.save_dir, f"eval_results_{self.lambda_val}_lambda_"+caption+".txt")
        with open(path, "w") as f:
            f.write(f"Lambda: {self.lambda_val}\n")
            f.write(f"Trained for: {nb_steps} steps\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v:.6f}\n")
        print(f"Results saved to {path}")