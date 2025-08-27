import torch
import torch.nn as nn

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