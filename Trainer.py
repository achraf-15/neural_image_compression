import math
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

class Trainer:
    def __init__(self, model, optimizer, train_loader, val_loader=None, rd_loss=None, lambda_val=0.005, scheduler=None,
                 max_steps=10000, resume=False, log_interval=None, img_interval=None, val_interval=None, 
                log_dir="runs/experiment", checkpoint_path="./checkpoints/checkpoint.pth", device="cuda"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader 
        if rd_loss is None:
            raise ValueError("You must provide a rate-distortion loss function (`rd_loss`)")
        self.rd_loss = rd_loss # if rd_loss not avaible: error 
        self.lambda_val = lambda_val
        self.device = device
        
        self.max_steps = max_steps
        self.step = 0
        self.train_iter = iter(train_loader)
        self.log_interval = log_interval if log_interval else int(self.max_steps/200)
        self.img_interval = img_interval if img_interval else int(self.max_steps/25)
        self.val_interval = val_interval if val_interval else int(self.max_steps/200)

        # Scheduler setup
        if scheduler == 'plateau': 
            self.scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=100, factor=0.5)
            self.use_plateau = True
        elif scheduler == 'cosine':
            self.scheduler = CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=1e-5)
            self.use_plateau = False
        else:
            self.scheduler = None
            self.use_plateau = False

        # Resume training
        self.resume = resume
        self.checkpoint_path = checkpoint_path
        if self.resume and self.checkpoint_path is not None and os.path.exists(self.checkpoint_path):
            self.load_checkpoint()

        # Writer: purge_step ensures continuation of logs
        self.writer = SummaryWriter(log_dir, purge_step=self.step)

    
    def save_checkpoint(self):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
        }
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        torch.save(checkpoint, self.checkpoint_path)
        print(f"Checkpoint saved at step {self.step} -> {self.checkpoint_path}")

    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler is not None and checkpoint["scheduler"] is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.step = checkpoint["step"]
        self.max_steps += self.step
        print(f"Checkpoint loaded -> Resuming from step {self.step}")

    def train(self):
        pbar = tqdm(total=self.max_steps, initial=self.step, desc="Training")

        while self.step < self.max_steps:
                
            imgs = self._next_batch()
            imgs = imgs.to(self.device)

            self.optimizer.zero_grad()
            model_out = self.model(imgs)
            results = self.rd_loss(model_out, imgs, self.lambda_val)

            results['loss'].backward()
            self.optimizer.step()

            # Logging scalars
            self._log_scalars(results)

            # Validation 
            if self.val_loader is not None and self.step % self.val_interval == 0:
                val_loss = self._validate()
                if self.use_plateau:
                    self.scheduler.step(val_loss)
    
            # Cosine annealing steps every iteration
            if self.scheduler is not None and not self.use_plateau:
                self.scheduler.step()

            # Check Learning rate decay
            if self.scheduler is not None:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar("train/learning_rate", current_lr, self.step)

            # Logging histograms
            if self.step % self.log_interval == 0:
                self._log_histograms(model_out)
                self._log_channel_activity(model_out, tensor_name='y')
                self._log_channel_activity(model_out, tensor_name='z')
                self._log_entropy_params(model_out)

            # Logging images
            if self.step % self.img_interval == 0:
                self._log_paired_images(imgs, model_out)
                self._log_entropy_heatmap(model_out, tensor_name='y')
                self._log_entropy_heatmap(model_out, tensor_name='z')
                self._log_latent_heatmap(model_out, tensor_name='y')
                self._log_latent_heatmap(model_out, tensor_name='z')
                self._log_entropy_cdf(model_out, tensor_name='z')

                
            self.step += 1
            pbar.update(1)

        pbar.close()
        self.writer.close()

        # save model after training
        if self.checkpoint_path is not None:
            self.save_checkpoint()

    def _next_batch(self):
        try:
            return next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_loader)
            return next(self.train_iter)

    def _log_scalars(self, results):
        for k, v in results.items():
            if isinstance(v, (float, int)):
                self.writer.add_scalar(f"losses/{k}", v, self.step)

    def _validate(self):
        self.model.eval()
        total_loss = 0
        bpp_loss = 0
        psnr_loss = 0
        with torch.no_grad():
            for imgs in self.val_loader:
                imgs = imgs.to(self.device)
                model_out = self.model(imgs, training=False)
                results = self.rd_loss(model_out, imgs, self.lambda_val)
                total_loss += results['loss'].item()
                bpp_loss += results['bpp_total']
                psnr_loss += results['psnr']
        self.model.train()
        avg_loss = total_loss / len(self.val_loader)
        avg_bpp = bpp_loss / len(self.val_loader)
        avg_psnr = psnr_loss / len(self.val_loader)
        self.writer.add_scalar("validation/validation_loss", avg_loss, self.step)
        self.writer.add_scalar("validation/validation_bpp", avg_bpp, self.step)
        self.writer.add_scalar("validation/validation_pnsr", avg_psnr, self.step)
        return avg_loss

    def _log_histograms(self, model_out):
        self.writer.add_histogram("latents/y", model_out['y'], self.step)
        self.writer.add_histogram("latents/y_hat", model_out['y_in'], self.step)
        self.writer.add_histogram("latents/z", model_out['z'], self.step)
        self.writer.add_histogram("latents/z_hat", model_out['z_in'], self.step)

        # Logp distributions (log-domain, more stable)
        self.writer.add_histogram("probability/logp_y", model_out['logp_y'], self.step)
        self.writer.add_histogram("probability/logp_z", model_out['logp_z'], self.step)

        # Distributions
        self.writer.add_histogram("probability/p_y", model_out['p_y'], self.step)
        self.writer.add_histogram("probability/p_z", model_out['p_z'], self.step)

        # Entropies 
        self.writer.add_histogram("entropy/y", -model_out['logp_y'] / math.log(2), self.step)
        self.writer.add_histogram("entropy/z", -model_out['logp_z'] / math.log(2), self.step)
        self.writer.add_histogram("entropy/y_per_component", -model_out['logp_y'].sum(dim=(2,3)) / math.log(2), self.step)
        self.writer.add_histogram("entropy/z_per_component", -model_out['logp_z'].sum(dim=(2,3)) / math.log(2), self.step)

        # Means for quick scalar tracking
        self.writer.add_scalar("probability/logp_y_mean", model_out['logp_y'].mean().item(), self.step)
        self.writer.add_scalar("probability/logp_z_mean", model_out['logp_z'].mean().item(), self.step)

        self.writer.add_scalar("probability/p_y_mean", model_out['p_y'].mean().item(), self.step)
        self.writer.add_scalar("probability/p_z_mean", model_out['p_z'].mean().item(), self.step)

        self.writer.add_scalar("entropy/entropy_y_mean", (-model_out['logp_y'] / math.log(2)).mean().item(), self.step)
        self.writer.add_scalar("entropy/entropy_z_mean", (-model_out['logp_z'] / math.log(2)).mean().item(), self.step)

    def _log_channel_activity(self, model_out, tensor_name='y'):
        logp = model_out['logp_' + tensor_name]
        
        # discrete entropy estimate (bits)
        avg_bits_per_c = (-logp / math.log(2.0)).mean(dim=(0,2,3)) # [C]
        dead_channels = (avg_bits_per_c < 1e-4 ).float().sum().item()
        self.writer.add_scalar(f"activity/{tensor_name}_dead_channels_by_entropy", dead_channels, self.step)

        
    def _log_entropy_params(self, model_out):
        # GaussianConditional
        if 'mu' in model_out and 'sigma' in model_out:
            self.writer.add_histogram("entropy_params/mu", model_out['mu'], self.step)
            self.writer.add_histogram("entropy_params/sigma", model_out['sigma'], self.step)
        # GaussianMixtureConditional
        if 'weights' in model_out:
            self.writer.add_histogram("entropy_params/weights", model_out['weights'], self.step)
            self.writer.add_histogram("entropy_params/mus", model_out['mus'], self.step)
            self.writer.add_histogram("entropy_params/sigmas", model_out['sigmas'], self.step)
            used_components = (model_out['weights'] > 1e-4).float().sum(dim=1).mean().item() # average number of used components per mixture
            self.writer.add_scalar("entropy_params/used_components_mean", used_components, self.step)

    def _log_paired_images(self, imgs, model_out, max_samples=4):
        imgs = imgs.detach().cpu()
        recon = model_out['x_hat'].detach().cpu()
        
        paired = []
        for i in range(min(max_samples, imgs.size(0))):
            paired.append(imgs[i])
            paired.append(recon[i])
        
        grid = vutils.make_grid(torch.stack(paired), nrow=2, normalize=True, scale_each=True)
        self.writer.add_image("comparison/paired", grid, self.step)


    def _log_entropy_heatmap(self, model_out, tensor_name='y'):
        x = model_out['logp_' + tensor_name]  # [B, C, H, W]
        h = model_out['logp_' + tensor_name]  # channel with highest entropy ( latent or hyperlatent )  
    
        max_idx = h[0].sum(dim=(1,2)).argmin()  # channel with highest total entropy for first image in batch (entropy = -logp)
    
        # per-pixel entropy in bits
        entropy_bits = (-x[0, max_idx] / math.log(2)).detach().cpu().numpy()
        entropy_bits = (entropy_bits - entropy_bits.min()) / (entropy_bits.max() - entropy_bits.min() + 1e-12)

        self.writer.add_image(f"heatmaps/qantized_{tensor_name}_entropy", entropy_bits[None, :, :], self.step, dataformats='CHW')

    def _log_latent_heatmap(self, model_out, tensor_name='y'):
        x = model_out[tensor_name]  # [B, C, H, W]
        h = model_out['logp_' + tensor_name]  # channel with highest entropy ( latent or hyperlatent )  
    
        max_idx = h[0].sum(dim=(1,2)).argmin()   # channel with highest total entropy for first image in batch (entropy = -logp)
    
        # Take first sample for visualization
        heatmap = x[0, max_idx].detach().cpu().numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-12)
        self.writer.add_image(f"heatmaps/latent_{tensor_name}_heatmap", heatmap[None, :, :], self.step, dataformats='CHW')

    def _log_entropy_cdf(self, model_out, tensor_name='z', num_points=200):
        
        if not hasattr(self.model, "factorized_entropy_model"):
            return

        logp = model_out[f'logp_{tensor_name}'][0]  # [C, H, W] for first sample
        entropy_per_channel = -logp.sum(dim=(1, 2)) / math.log(2)  # [C]
        entropy_sorted, indices = torch.sort(entropy_per_channel)

        # pick lowest, median, highest entropy channels
        low_idx = indices[0].item()
        mid_idx = indices[len(indices) // 2].item()
        high_idx = indices[-1].item()
        selected_channels = [low_idx, mid_idx, high_idx]

        z = model_out[tensor_name][0]   # [C, H, W]
        min_val = z.min().item()
        max_val = z.max().item()
        margin = 3 * z.std().item()
        value_range = (min_val - margin, max_val + margin)
        
        # values over which to evaluate CDF
        xs = torch.linspace(value_range[0], value_range[1], num_points).to(self.device)

        # FactorizedEntropyBottleneck
        EP = self.model.factorized_entropy_model  
        
        with torch.no_grad():
            xs_np = xs.detach().cpu().numpy()
            cdf_curves = []
            pmf_cont_curves = []
            pmf_disc_curves = []
            z_ranges = []
            labels = []
            for ch in selected_channels:
                cdf = EP.channel_cdf(ch, xs)      # [N]
                pmf_cont = EP.channel_pmf(ch, xs)      # [N]
                cdf_curves.append(cdf.detach().cpu().numpy())
                pmf_cont_curves.append(pmf_cont.detach().cpu().numpy())

                # discrete PMF: evaluate only at integer positions inside range
                xs_disc = torch.arange(int(value_range[0]), int(value_range[1])+1, device=self.device)
                pmf_disc = EP.channel_pmf(ch, xs_disc)
                pmf_disc_curves.append((xs_disc.detach().cpu().numpy(), pmf_disc.detach().cpu().numpy()))

                # z-range for transparent box
                z_ch = z[ch].flatten().cpu().numpy()
                z_ranges.append((z_ch.min(), z_ch.max()))
                
                labels.append(f"ch {ch} ({entropy_per_channel[ch].item():.2f} bits)")

         # Use default matplotlib colors
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # ---- CDF figure ----
        fig_cdf, ax = plt.subplots(1, 1, figsize=(6, 3.5))
        for i, (y, (z_min, z_max), lab) in enumerate(zip(cdf_curves, z_ranges, labels)):
            color = colors[i % len(colors)]
            # transparent box for actual z range
            ax.axvspan(z_min, z_max, alpha=0.15, color=color)
            ax.plot(xs_np, y, linewidth=1.5, label=lab, color=color)
        ax.set_title("Factorized bottleneck CDF (per channel)")
        ax.set_xlabel("x")
        ax.set_ylabel("CDF(x)")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize=8)
        self.writer.add_figure("bottleneck/cdf", fig_cdf, global_step=self.step)
        plt.close(fig_cdf)
    
        # ---- PMF figure ----
        fig_pmf, ax = plt.subplots(1, 1, figsize=(6, 3.5))
        for i, (pmf_cont, (xs_d, pmf_d), (z_min, z_max), lab) in enumerate(zip(
            pmf_cont_curves, pmf_disc_curves, z_ranges, labels)
        ):
            color = colors[i % len(colors)]
            # transparent box for actual z range
            ax.axvspan(z_min, z_max, alpha=0.15, color=color)
            # discrete PMF: vertical lines + dots
            ax.vlines(xs_d, 0, pmf_d, colors=color, linestyles='dotted', alpha=0.6, linewidth=0.7)
            ax.plot(xs_d, pmf_d, 'o', color=color, markersize=3, alpha=0.8)
            # continuous PMF
            ax.plot(xs_np, pmf_cont, linewidth=1.5, label=lab, color=color)
    
        ax.set_title("Factorized bottleneck PMF (continuous + discrete)")
        ax.set_xlabel("x")
        ax.set_ylabel("PMF(bin around x)")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize=8)
        self.writer.add_figure("bottleneck/pmf", fig_pmf, global_step=self.step)
        plt.close(fig_pmf)

