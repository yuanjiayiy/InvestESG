from env.pettingzoo.investesg import InvestESG
import wandb
from stable_baselines3.common.policies import ActorCriticPolicy
import torch

class HardCodePolicy:
    def __init__(self, policy, random=False, action=None, **kwargs):
        self.random = random
        self.policy = policy
        self.action = action
    
    def set_training_mode(self, mode):
        self.policy.set_training_mode(mode)

    def forward(self, obs):
        if self.random:
            action = self.policy.action_space.sample()
            return torch.Tensor([action]), None, None
        elif self.action:
            return torch.Tensor([self.action]), None, None
        return None, None, None