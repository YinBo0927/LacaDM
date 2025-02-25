import torch
import numpy as np
import torch.nn as nn

class GaussianDiffusion:
    """Implements continuous diffusion process"""
    def __init__(self, config):
        self.timesteps = config.timesteps
        self.beta = self._linear_beta_schedule()
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)
        
    def _linear_beta_schedule(self):
        return torch.linspace(1e-4, 0.02, self.timesteps)
    
    def q_sample(self, x0, t, noise=None):
        """Forward diffusion process"""
        if noise is None:
            noise = torch.randn_like(x0)
            
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t].view(-1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].view(-1, 1)
        
        return sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
    
    def p_losses(self, model, x0, t):
        """Calculate loss for reverse process"""
        noise = torch.randn_like(x0)
        x_noisy = self.q_sample(x0, t, noise)
        predicted_noise = model(x_noisy, t)
        return torch.mean((noise - predicted_noise)**2)

class BernoulliDiffusion:
    """Implements discrete diffusion process"""
    def __init__(self, config):
        self.timesteps = config.timesteps
        self.beta = self._linear_beta_schedule()
        self.gamma = torch.cumprod(1 - self.beta, dim=0)
        
    def _linear_beta_schedule(self):
        return torch.linspace(1e-4, 0.02, self.timesteps)
    
    def q_sample(self, x0, t):
        """Forward diffusion for discrete data"""
        gamma_t = self.gamma[t].view(-1, 1)
        return torch.bernoulli(gamma_t * x0 + (1 - gamma_t)/2)
    
    def p_losses(self, model, x0, t):
        """Discrete cross-entropy loss"""
        x_noisy = self.q_sample(x0, t)
        logits = model(x_noisy, t)
        return nn.BCEWithLogitsLoss()(logits, x0)