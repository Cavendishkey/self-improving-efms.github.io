import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from dotenv import load_dotenv
import os

from models.policy_net import PolicyNet
from models.steps_net import StepsNet
from env.point_mass import Point2D, pd_controller

# Load configuration from .env file
load_dotenv()

ACTION_SCALE = float(os.getenv("ACTION_SCALE", 0.001))
DEVICE = os.getenv("DEVICE", "cpu")
MAX_STEPS = int(os.getenv("MAX_STEPS", 200))
NUM_EXPERT_EPISODES = int(os.getenv("NUM_EXPERT_EPISODES", 1000))
SFT_EPOCHS = int(os.getenv("SFT_EPOCHS", 50))
SFT_LR = float(os.getenv("SFT_LR", 3e-4))
SFT_BATCH_SIZE = int(os.getenv("SFT_BATCH_SIZE", 256))
SFT_EVAL_EPISODES = int(os.getenv("SFT_EVAL_EPISODES", 200))


def collect_expert_data(num_episodes=NUM_EXPERT_EPISODES, max_steps=MAX_STEPS):
    env = Point2D()
    all_obs = []
    all_actions = []
    all_steps = []
    successes = 0

    for ep in range(num_episodes):
        ts = env.reset()
        obs = ts.observation
        ep_len = 0
        ep_obs = []
        ep_actions = []

        for t in range(max_steps):
            cur_pos = obs['cur_pos']
            cur_vel = obs['cur_vel']
            goal_pos = obs['goal_pos']

            action = pd_controller(cur_pos, cur_vel, goal_pos)
            action = np.clip(action, -1.0, 1.0)

            obs_vec = np.concatenate([cur_pos, cur_vel, goal_pos])
            ep_obs.append(obs_vec)
            ep_actions.append(action / ACTION_SCALE)

            ts = env.step(action)
            obs = ts.observation
            ep_len += 1

            if env.success():
                successes += 1
                break

        for i, (o, a) in enumerate(zip(ep_obs, ep_actions)):
            remaining = ep_len - 1 - i
            remaining = max(0, min(remaining, 199))
            all_obs.append(o)
            all_actions.append(a)
            all_steps.append(remaining)

    print(f"Expert success rate: {successes/num_episodes:.2f} ({successes}/{num_episodes})")
    return np.array(all_obs), np.array(all_actions), np.array(all_steps)


def train_sft(obs, actions, steps_labels,
              num_epochs=SFT_EPOCHS, batch_size=SFT_BATCH_SIZE,
              lr=SFT_LR, device=DEVICE):
    obs_t = torch.tensor(obs, dtype=torch.float32).to(device)
    acts_t = torch.tensor(actions, dtype=torch.float32).to(device)

    steps_labels = np.clip(steps_labels, 0, 199)
    steps_t = torch.tensor(steps_labels, dtype=torch.long).to(device)

    dataset = TensorDataset(obs_t, acts_t, steps_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    policy = PolicyNet().to(device)
    steps_net = StepsNet().to(device)
    optimizer = optim.Adam(list(policy.parameters()) + list(steps_net.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    writer = SummaryWriter("runs/sft_training")

    for epoch in range(num_epochs):
        total_policy_loss = 0.0
        total_steps_loss = 0.0
        for batch_obs, batch_act, batch_steps in loader:
            mean, std = policy(batch_obs)
            logits = steps_net(batch_obs)

            dist = torch.distributions.Normal(mean, std)
            log_prob = dist.log_prob(batch_act).sum(dim=-1)
            policy_loss = -log_prob.mean()

            steps_loss = nn.CrossEntropyLoss()(logits, batch_steps)

            loss = policy_loss + steps_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_steps_loss += steps_loss.item()

        scheduler.step()
        n = len(loader)
        avg_policy_loss = total_policy_loss / n
        avg_steps_loss = total_steps_loss / n

        # ===== TensorBoard logging =====
        writer.add_scalar("sft/policy_loss", avg_policy_loss, epoch + 1)
        writer.add_scalar("sft/steps_loss", avg_steps_loss, epoch + 1)
        writer.add_scalar("sft/total_loss", avg_policy_loss + avg_steps_loss, epoch + 1)
        writer.add_scalar("sft/learning_rate", scheduler.get_last_lr()[0], epoch + 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, "
                  f"Policy Loss: {avg_policy_loss:.4f}, "
                  f"Steps Loss: {avg_steps_loss:.4f}, "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")

    writer.close()
    print("TensorBoard logs saved to runs/sft_training/")

    torch.save(policy.state_dict(), 'policy_sft.pth')
    torch.save(steps_net.state_dict(), 'steps_net_sft.pth')
    print("Models saved.")
    return policy, steps_net


def evaluate_sft(policy, num_episodes=SFT_EVAL_EPISODES,
                 max_steps=MAX_STEPS, device=DEVICE):
    env = Point2D()
    successes = 0
    total_len = 0
    policy.eval()

    for _ in range(num_episodes):
        ts = env.reset()
        obs = ts.observation

        for t in range(max_steps):
            obs_vec = np.concatenate([obs['cur_pos'], obs['cur_vel'], obs['goal_pos']])
            obs_t = torch.tensor(obs_vec, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                mean, std = policy(obs_t)
                action = mean.squeeze(0).cpu().numpy() * ACTION_SCALE

            action = np.clip(action, -1.0, 1.0)
            ts = env.step(action)
            obs = ts.observation

            if env.success():
                successes += 1
                total_len += t + 1
                break
        else:
            total_len += max_steps

    sr = successes / num_episodes
    avg_len = total_len / num_episodes
    print(f"SFT Eval - Success Rate: {sr:.2f}, Avg Length: {avg_len:.1f}")
    policy.train()
    return sr


def main():
    print(f"Configuration: ACTION_SCALE={ACTION_SCALE}, DEVICE={DEVICE}, "
          f"SFT_EPOCHS={SFT_EPOCHS}, SFT_LR={SFT_LR}, SFT_BATCH_SIZE={SFT_BATCH_SIZE}")

    print("=" * 50)
    print("Collecting expert data...")
    print("=" * 50)
    obs, actions, steps_labels = collect_expert_data()
    print(f"Collected {len(obs)} samples, "
          f"scaled action range: [{actions.min():.3f}, {actions.max():.3f}]")

    print("\n" + "=" * 50)
    print(f"Training SFT ({SFT_EPOCHS} epochs)...")
    print("=" * 50)
    policy, steps_net = train_sft(obs, actions, steps_labels)

    print("\n" + "=" * 50)
    print("Evaluating SFT policy...")
    print("=" * 50)
    sr = evaluate_sft(policy)

    print(f"\n✓ Stage 1 complete! ACTION_SCALE = {ACTION_SCALE}")


if __name__ == "__main__":
    main()
