import torch
from tqdm import tqdm
import torch.nn.functional as F

class LaCaDMTrainer:
    def train_step(self, batch):
        states, actions, rewards = batch
        self.optimizer.zero_grad()
        
        x = torch.cat([states, actions, rewards], dim=-1)
        t = torch.randint(0, self.config.timesteps, (x.size(0),), 
                        device=self.config.device)
        
        # Forward pass with VAE-LCL
        pred_actions, mu, logvar, kl_loss = self.model(x, t)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(pred_actions, actions)
        
        # KL divergence
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        loss = recon_loss + self.config.beta * kl_div + kl_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                     self.config.grad_clip)
        self.optimizer.step()
        
        return {
            'total_loss': loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_div': kl_div.item(),
            'lcl_loss': kl_loss.item()
        }