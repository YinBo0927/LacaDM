import torch
import torch.nn as nn
from .utils import Swish
from .lcl import VAE_LCL

class PolicyEncoder(nn.Module):
    """Encodes state-action-reward sequences into latent space"""
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, latent_dim*2)
        )
        
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.mlp(h_n[-1])

class PolicyDecoder(nn.Module):
    """Decodes latent vectors into policies"""
    def __init__(self, output_dim, latent_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, z):
        return self.mlp(z)

# lacadm/model.py
class LaCaDM(nn.Module):
    def __init__(self, state_dim, action_dim, reward_dim, config):
        super().__init__()
        self.config = config
        
        # VAE Encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim+action_dim+reward_dim, 256),
            Swish(),
            nn.Linear(256, 128),
            Swish(),
            nn.Linear(128, config.latent_dim*2)  # mu and logvar
        )
        
        # LCL Module
        self.lcl = VAE_LCL(config.latent_dim, 
                          config.hidden_dim,
                          num_causal_factors=8)
        
        # Decoder with time conditioning
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim + 1, 256),  # +1 for time embedding
            Swish(),
            nn.Linear(256, action_dim)
        )
        
    def forward(self, x, t):
        # VAE Encoding
        mu_logvar = self.encoder(x)
        mu, logvar = torch.chunk(mu_logvar, 2, dim=-1)
        z = self.reparameterize(mu, logvar)
        
        # Time embedding
        t_emb = t.float() / self.config.timesteps
        
        # LCL causal transition
        z_prev = z[:-1]  # previous states
        z_current = z[1:]  # current states
        z_pred, kl_loss = self.lcl(z_prev, z_current)
        
        # Decoding with time
        z_time = torch.cat([z, t_emb.unsqueeze(-1)], dim=-1)
        actions = self.decoder(z_time)
        
        return actions, mu, logvar, kl_loss