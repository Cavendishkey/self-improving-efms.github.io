import torch.nn as nn

class StepsNet(nn.Module):
    def __init__(self, obs_dim=6, num_steps=200, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_steps)  # 输出 logits
        )

    def forward(self, obs):
        return self.net(obs)
