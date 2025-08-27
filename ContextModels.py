import torch.nn as nn

class MaskedConv2d(nn.Conv2d): # taken from Github 
    """
    Implementation of the Masked convolution from the paper
    Van den Oord, Aaron, et al. "Conditional image generation with pixelcnn decoders."
    https://arxiv.org/pdf/1606.05328.pdf
    """
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in ('A', 'B')
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class ContextModel(nn.Module): # taken from Github 
    def __init__(self, latent_channels=192):
        super(ContextModel, self).__init__()
        self.masked = MaskedConv2d(
            "A",
            in_channels=latent_channels,
            out_channels=2 * latent_channels,
            kernel_size=5,
            stride=1,
            padding=2
        )

    def forward(self, x):
        return self.masked(x)