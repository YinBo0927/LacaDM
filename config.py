import torch

class Config:
    # Diffusion parameters
    timesteps = 1500
    beta_schedule = 'linear'
    
    # Model architecture
    latent_dim = 128
    hidden_dim = 256
    num_layers = 4
    
    # Training
    lr = 1e-4
    batch_size = 64
    epochs = 1000
    grad_clip = 1.0
    
    # VAE parameters
    beta = 0.5  # KL loss weight
    num_causal_factors = 8  # LCL causal factors
    
    # LCL parameters 
    lcl_lambda = 0.1  # causal loss weight
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")