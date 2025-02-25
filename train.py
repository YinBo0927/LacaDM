from lacadm import LaCaDM, GaussianDiffusion, BernoulliDiffusion, LaCaDMTrainer
from lacadm.utils import create_optimizer
from config import Config
import gymnasium as gym
import mogymnasium as mogym
from torch.utils.data import DataLoader
import torch
import numpy as np

class PCNCollector:
    def sample_trajectory(self, weights):
        self.env.set_preference(weights)
        
        states, actions, rewards = [], [], []
        state, _ = self.env.reset()
        
        for _ in range(self.max_steps):
            action = self.policy(state, weights)
            
            next_state, vector_reward, terminated, truncated, _ = self.env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(vector_reward)
            
            if terminated or truncated:
                break
            state = next_state
        
        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards)
        }

def prepare_dataset(env_name):
    env = mogym.make(
        env_name,
        render_mode=None, 
        reward_weights=np.array([0.5, 0.5])  
    )
    
    collector = PCNCollector(env)
    
    trajectories = []
    for _ in range(1000): 
        weights = np.random.dirichlet([1]*env.reward_dim)
        traj = collector.sample_trajectory(weights)
        trajectories.append(traj)
    
    dataset = {
        'states': torch.FloatTensor([s for traj in trajectories for s in traj['states']]),
        'actions': torch.FloatTensor([a for traj in trajectories for a in traj['actions']]),
        'rewards': torch.FloatTensor([r for traj in trajectories for r in traj['rewards']])
    }
    
    return DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)

def main():
    config = Config()
    
    # Initialize model and diffusion(need to modify according to the MORL task)
    model = LaCaDM(10, 4, 2, config)
    diffusion = GaussianDiffusion(config)  # or BernoulliDiffusion
    
    optimizer = create_optimizer(model, config)
    trainer = LaCaDMTrainer(model, diffusion, optimizer, config)
    
    # Load dataset
    train_loader = prepare_dataset("DeepSeaTreasure-v0") # modify the environment name
    
    # Train model
    trainer.train(train_loader, config.epochs)

if __name__ == "__main__":
    main()