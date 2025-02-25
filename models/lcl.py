import torch
import torch.nn as nn
from .utils import Swish

class VAE_LCL(nn.Module):
    """VAE-based Latent Causal Learning module"""
    def __init__(self, latent_dim, hidden_dim, num_causal_factors):
        super().__init__()
        # Causal discovery layers
        self.fc_mu = nn.Linear(latent_dim, num_causal_factors)
        self.fc_logvar = nn.Linear(latent_dim, num_causal_factors)
        
        # Causal inference network
        self.causal_mlp = nn.Sequential(
            nn.Linear(num_causal_factors, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def forward(self, z_prev, z_current):
        """Implements causal VAE structure"""
        # Causal factor extraction
        mu = self.fc_mu(z_prev)
        logvar = self.fc_logvar(z_prev)
        c = self.reparameterize(mu, logvar)
        
        # Causal transition
        z_pred = self.causal_mlp(c)
        
        # KL divergence for VAE
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return z_pred, kl_loss

class LCLLoss(nn.Module):
    """Combined VAE and causal loss"""
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
        self.mse = nn.MSELoss()
        
    def forward(self, z_pred, z_true, kl_loss):
        recon_loss = self.mse(z_pred, z_true)
        total_loss = recon_loss + self.beta * kl_loss
        return total_loss, (recon_loss.item(), kl_loss.item())