import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class FirstHalf(nn.Module):
    def __init__(self, model, cut_layer):
        super().__init__()
        self.model = model.model.model[:cut_layer+1]  # keep first N layers

    def forward(self, x):
        outputs = {}
        for i, m in enumerate(self.model):
            if m.f != -1:
                if isinstance(m.f, int):
                    x = outputs[m.f]
                else:
                    x = torch.cat([outputs[j] for j in m.f], 1)
            x = m(x)
            outputs[i] = x
        return x  # final tensor

class SecondHalf(nn.Module):
    def __init__(self, model, cut_layer):
        super().__init__()
        self.model = model
        self.cut = cut_layer

    def forward(self, x, prev_outputs=None):
        outputs = {} if prev_outputs is None else dict(prev_outputs)
        outputs[self.cut] = x
        z = x
        for i, m in list(enumerate(self.model.model.model))[self.cut+1:]:
            if m.f != -1:
                if isinstance(m.f, int):
                    #print(i,i+m.f)
                    z = outputs[i+m.f]
                else:
                    #print([self.cut+1+i-j for j in m.f]) 
                    z = [outputs[i-1]] + [outputs[j] for j in m.f[1:]]
            z = m(z)
            outputs[i] = z
            #print(outputs.keys())
        return z

class FrozenActivationBlock(nn.Module):
    def __init__(self, model, cut_layer: int):
        super().__init__()
        bn = model.model.model[cut_layer].bn

        self.frozen_bn = nn.BatchNorm2d(
            num_features=bn.num_features,
            eps=bn.eps,
            momentum=bn.momentum,
            affine=True,
            track_running_stats=True
        )

        with torch.no_grad():
            self.frozen_bn.weight.copy_(bn.weight)
            self.frozen_bn.bias.copy_(bn.bias)
            self.frozen_bn.running_mean.copy_(bn.running_mean)
            self.frozen_bn.running_var.copy_(bn.running_var)

        for p in self.frozen_bn.parameters():
            p.requires_grad = False

        self.frozen_bn.eval()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.frozen_bn(x))
        


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
    ax1.plot(steps, total_bpp, color=color, label="Total Latent Information", linewidth=2)
    ax1.tick_params(axis="y", labelcolor=color)
    
    # right axis (percentage)
    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Vision Task Information (%)", color=color)
    ax2.plot(steps, vision_info, color=color, linestyle="--", label="Vision Task Information (%)", linewidth=2)
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
    plt.plot(steps, values, color="tab:blue", linewidth=2)
    plt.xlabel("Training Steps")
    plt.ylabel(y_label)
    plt.title(f"Evolution of {y_label} Through Training")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
