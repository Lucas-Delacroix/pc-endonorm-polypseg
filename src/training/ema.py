import copy
import torch
import torch.nn as nn

class ModelEMA:

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        ema_parameters = dict(self.module.named_parameters())
        for name, parameter in model.named_parameters():
            ema_parameters[name].mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)
        ema_buffers = dict(self.module.named_buffers())
        for name, buffer in model.named_buffers():
            ema_buffers[name].copy_(buffer)

    def to(self, device):
        self.module.to(device)
        return self
