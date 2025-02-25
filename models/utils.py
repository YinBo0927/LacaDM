import torch
import torch.nn as nn

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def create_optimizer(model, config):
    return torch.optim.AdamW(model.parameters(), lr=config.lr)