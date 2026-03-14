from env.point_mass import Point2D
from models.policy_net import PolicyNet
from models.steps_net import StepsNet
from utils.reinforce import reinforce_train, evaluate_policy
import matplotlib.pyplot as plt
import torch
from dotenv import load_dotenv
import os

# Load configuration from .env file
load_dotenv()

ACTION_SCALE = float(os.getenv("ACTION_SCALE", 0.001))
DEVICE = os.getenv("DEVICE", "cpu")
RL_ITERATIONS = int(os.getenv("RL_ITERATIONS", 100))
RL_EPISODES_PER_ITER = int(os.getenv("RL_EPISODES_PER_ITER", 20))
RL_LR = float(os.getenv("RL_LR", 1e-4))
RL_DISCOUNT = float(os.getenv("RL_DISCOUNT", 0.99))
RL_EVAL_EPISODES = int(os.getenv("RL_EVAL_EPISODES", 50))


def main():
    device = torch.device(DEVICE)
    print(f"Using device: {device}")
    print(f"Configuration: RL_ITERATIONS={RL_ITERATIONS}, RL_LR={RL_LR}, "
          f"RL_EPISODES_PER_ITER={RL_EPISODES_PER_ITER}, RL_DISCOUNT={RL_DISCOUNT}")

    env = Point2D()

    policy = PolicyNet().to(device)
    steps_net = StepsNet().to(device)

    policy.load_state_dict(torch.load('policy_sft.pth', map_location=device, weights_only=True))
    steps_net.load_state_dict(torch.load('steps_net_sft.pth', map_location=device, weights_only=True))
    print("Loaded Stage 1 models.")

    # Evaluate Stage 1 baseline
    sft_success, sft_len_mean, sft_len_std = evaluate_policy(
        env, policy, steps_net, num_episodes=RL_EVAL_EPISODES, device=device,
        action_scale=ACTION_SCALE
    )
    print(f"Stage 1 Policy - Success Rate: {sft_success:.2f}, "
          f"Mean Length: {sft_len_mean:.1f} ± {sft_len_std:.1f}")

    # Stage 2 REINFORCE
    print("\nStarting Stage 2 Self-Improvement...")
    success_rates = reinforce_train(
        env, policy, steps_net,
        num_iterations=RL_ITERATIONS,
        episodes_per_iter=RL_EPISODES_PER_ITER,
        lr=RL_LR,
        gamma=RL_DISCOUNT,
        device=device,
        action_scale=ACTION_SCALE
    )

    # Plot Figure 4
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(success_rates) + 1), success_rates,
             marker='o', markersize=3, label='Stage 2 Self-Improvement')
    plt.axhline(y=sft_success, color='r', linestyle='--', label='Stage 1 SFT Baseline')
    plt.xlabel('REINFORCE Iteration')
    plt.ylabel('Success Rate')
    plt.title('Figure 4: Self-Improvement improves success rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    plt.savefig('figure4_reproduced.png', dpi=150)
    plt.show()
    print("Training complete. Figure saved as figure4_reproduced.png")


if __name__ == "__main__":
    main()
