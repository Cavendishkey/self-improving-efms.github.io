import torch
import torch.nn as nn

class PolicyNet(nn.Module):
    def __init__(self, obs_dim=6, act_dim=2, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, act_dim)
        # 初始化 log_std 为 -1，对应 std≈0.37
        # 归一化后的专家动作范围约 ±0.4，所以 std=0.37 是合理起点
        self.log_std = nn.Parameter(torch.full((act_dim,), -1.0))

    def forward(self, obs):
        features = self.net(obs)
        mean = self.mean_head(features)
        std = torch.exp(self.log_std.clamp(-5, 2))
        return mean, std
