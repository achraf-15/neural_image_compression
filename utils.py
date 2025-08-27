import torch
import matplotlib.pyplot as plt
import math


def gaussian_cdf(x: torch.Tensor):
    """Standard normal CDF via erf"""
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))



def plot_information_evolution(H_y, H_y1): #AI Generated 
    """
    Plot evolution of total latent rate (H_y) and vision task information ratio (H_y1/H_y).
    
    Args:
        H_y (list of tuples): [(step, value)] for total bpp
        H_y1 (list of tuples): [(step, value)] for base latent bpp
    """
    # unpack
    steps, total_bpp = zip(*H_y)
    _, base_bpp = zip(*H_y1)
    
    # compute ratio (%)
    vision_info = [b / t * 100 if t > 0 else 0 for b, t in zip(base_bpp, total_bpp)]
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    # left axis (total bpp)
    color = "tab:blue"
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Total Latent Information (bpp)", color=color)
    ax1.plot(steps, total_bpp, color=color, label="Total Latent Information", linewidth=0.5)
    ax1.tick_params(axis="y", labelcolor=color)
    
    # right axis (percentage)
    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Vision Task Information (%)", color=color)
    ax2.plot(steps, vision_info, color=color, linestyle="--", label="Vision Task Information (%)", linewidth=0.5)
    ax2.tick_params(axis="y", labelcolor=color)
    
    # title and layout
    plt.title("Evolution of Vision Task Information and Total Rate Through Training")
    fig.tight_layout()
    
    # optional legend (from both axes)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    
    plt.show()


def plot_metric_evolution(metric_list, y_label="Metric"): #AI Generated
    """
    Plot evolution of a single metric through training.
    
    Args:
        metric_list (list of tuples): [(step, value)]
        y_label (str): Label for the y-axis (also used in the plot title)
    """
    steps, values = zip(*metric_list)
    
    plt.figure(figsize=(8, 5))
    plt.plot(steps, values, color="tab:blue", linewidth=0.5)
    plt.xlabel("Training Steps")
    plt.ylabel(y_label)
    plt.title(f"Evolution of {y_label} Through Training")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
