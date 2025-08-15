import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import os
from pytorch_msssim import ms_ssim

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
        
        latents = out["y"][0]  # [C,H,W]
        entropy = out["logp_y"][0]  # [C,H,W]
        mean_entropy_per_channel = entropy.view(latents.shape[0], -1).mean(dim=1)
        
        # Pick highest entropy channel
        high_c = torch.argmax(mean_entropy_per_channel).item()
        
        # Case 1: Mean-Scale Gaussian (K=1)
        if "mu" in out and "sigma" in out:
            mean = out["mu"][0, high_c]
            scale = out["sigma"][0, high_c]
            latent = latents[high_c]
            pred_error = latent - mean
            latent_entropy = entropy[high_c]
            hyper_entropy = out["logp_z"][0, high_c]
            
            maps = [
                ("Original", img_np),
                ("Latent", latent.cpu().numpy()),
                ("Predicted Mean", mean.cpu().numpy()),
                ("Prediction Error", pred_error.cpu().numpy()),
                ("Predicted Scale", scale.cpu().numpy()),
                ("Latent Entropy", latent_entropy.cpu().numpy()),
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
        elif "weights" in out and "mus" in out and "sigmas" in out:
            weights = out["weights"][0, :, high_c]     # [K,H,W]
            mus = out["mus"][0, :, high_c]             # [K,H,W]
            sigmas = out["sigmas"][0, :, high_c]       # [K,H,W]
            latent = latents[high_c]                # [H,W]
            latent_entropy = -entropy[high_c] / math.log(2.0)
            hyper_entropy = -out["logp_z"][0, high_c] / math.log(2.0)
            K = mus.shape[0]
            
            # One row per component
            fig, axes = plt.subplots(K+1, 6, figsize=(18, 3*(K+1)))
            for k in range(K):
                pred_error = latent - mus[k]
                maps = [
                    (f"Comp {k} Weight", weights[k].cpu().numpy()),
                    (f"Comp {k} Mean", mus[k].cpu().numpy()),
                    (f"Comp {k} Pred Error", pred_error.cpu().numpy()),
                    (f"Comp {k} Sigma", sigmas[k].cpu().numpy()),
                    ("Latent Entropy", latent_entropy.cpu().numpy()),
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
            maps = [
                ("Original", img_np),
                ("Mixture Mean", mixture_mean.cpu().numpy()),
                ("Mixture Sigma", mixture_sigma.cpu().numpy()),
                ("Mixture Error", mixture_error.cpu().numpy()),
                ("Latent Entropy", latent_entropy.cpu().numpy()),
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