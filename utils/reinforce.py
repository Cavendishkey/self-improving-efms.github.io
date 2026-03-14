import numpy as np
import torch
import torch.optim as optim

def evaluate_policy(env, policy, steps_net, num_episodes=50,
                    max_steps=200, device='cpu', action_scale=0.001):
    """评估策略成功率"""
    policy.eval()
    successes = 0
    lengths = []

    for _ in range(num_episodes):
        ts = env.reset()
        obs = ts.observation

        for t in range(max_steps):
            obs_vec = np.concatenate([obs['cur_pos'], obs['cur_vel'], obs['goal_pos']])
            obs_t = torch.tensor(obs_vec, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                mean, std = policy(obs_t)
                action = mean.squeeze(0).cpu().numpy() * action_scale

            action = np.clip(action, -1.0, 1.0)
            ts = env.step(action)
            obs = ts.observation

            if env.success():
                successes += 1
                lengths.append(t + 1)
                break
        else:
            lengths.append(max_steps)

    policy.train()
    success_rate = successes / num_episodes
    mean_len = np.mean(lengths)
    std_len = np.std(lengths)
    return success_rate, mean_len, std_len


def collect_rollouts(env, policy, steps_net, num_episodes=10,
                     max_steps=200, device='cpu', action_scale=0.001):
    """收集 rollout 数据，用于 REINFORCE"""
    all_log_probs = []
    all_rewards = []
    all_intrinsic_rewards = []
    successes = 0

    policy.eval()
    steps_net.eval()

    for _ in range(num_episodes):
        ts = env.reset()
        obs = ts.observation

        ep_log_probs = []
        ep_rewards = []
        ep_intrinsic = []

        for t in range(max_steps):
            obs_vec = np.concatenate([obs['cur_pos'], obs['cur_vel'], obs['goal_pos']])
            obs_t = torch.tensor(obs_vec, dtype=torch.float32, device=device).unsqueeze(0)

            mean, std = policy(obs_t)
            dist = torch.distributions.Normal(mean, std)
            # 在归一化空间采样
            scaled_action = dist.sample()
            log_prob = dist.log_prob(scaled_action).sum(dim=-1)

            # 实际环境动作 = 归一化动作 × ACTION_SCALE
            action = scaled_action.squeeze(0).detach().cpu().numpy() * action_scale
            action = np.clip(action, -1.0, 1.0)

            # StepsNet 预测当前状态的剩余步数
            with torch.no_grad():
                logits_before = steps_net(obs_t)
                d_before = torch.argmax(logits_before, dim=-1).item()

            ts = env.step(action)
            obs = ts.observation

            # StepsNet 预测下一状态的剩余步数
            next_obs_vec = np.concatenate([obs['cur_pos'], obs['cur_vel'], obs['goal_pos']])
            next_obs_t = torch.tensor(next_obs_vec, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits_after = steps_net(next_obs_t)
                d_after = torch.argmax(logits_after, dim=-1).item()

            # 内在奖励：d_before - d_after（进步越多，奖励越高）
            intrinsic_reward = float(d_before - d_after)

            # 环境奖励
            env_reward = ts.reward if ts.reward is not None else 0.0

            ep_log_probs.append(log_prob)
            ep_rewards.append(env_reward)
            ep_intrinsic.append(intrinsic_reward)

            if env.success():
                successes += 1
                # 成功奖励
                ep_rewards[-1] += 10.0
                break

        all_log_probs.append(ep_log_probs)
        all_rewards.append(ep_rewards)
        all_intrinsic_rewards.append(ep_intrinsic)

    policy.train()
    steps_net.train()
    success_rate = successes / num_episodes
    return all_log_probs, all_rewards, all_intrinsic_rewards, success_rate


def compute_returns(rewards, intrinsic_rewards, gamma=0.99, intrinsic_weight=0.5):
    """计算带内在奖励的折扣回报"""
    combined = [r + intrinsic_weight * ir for r, ir in zip(rewards, intrinsic_rewards)]
    returns = []
    G = 0
    for r in reversed(combined):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def reinforce_train(env, policy, steps_net,
                    num_iterations=100, episodes_per_iter=20,
                    lr=1e-4, gamma=0.99, device='cpu',
                    action_scale=0.001):
    """REINFORCE + 内在奖励的自我改进训练"""
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    success_rates = []

    for iteration in range(1, num_iterations + 1):
        # 收集 rollout
        all_log_probs, all_rewards, all_intrinsic, sr = collect_rollouts(
            env, policy, steps_net,
            num_episodes=episodes_per_iter,
            device=device,
            action_scale=action_scale
        )

        # 计算 policy loss
        policy_loss = torch.tensor(0.0, device=device)
        total_episodes = 0

        for ep_log_probs, ep_rewards, ep_intrinsic in zip(all_log_probs, all_rewards, all_intrinsic):
            returns = compute_returns(ep_rewards, ep_intrinsic, gamma=gamma)
            returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

            # 标准化回报（减少方差）
            if len(returns_t) > 1:
                returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

            for log_prob, G in zip(ep_log_probs, returns_t):
                policy_loss -= log_prob.squeeze() * G

            total_episodes += 1

        policy_loss = policy_loss / total_episodes

        optimizer.zero_grad()
        policy_loss.backward()
        # 梯度裁剪，防止更新过大
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        success_rates.append(sr)

        # 每10次迭代做一次更准确的评估
        if iteration % 10 == 0:
            eval_sr, eval_len, _ = evaluate_policy(
                env, policy, steps_net,
                num_episodes=50, device=device,
                action_scale=action_scale
            )
            print(f"Iter {iteration}/{num_iterations}, "
                  f"Train SR: {sr:.2f}, Eval SR: {eval_sr:.2f}, "
                  f"Eval Len: {eval_len:.1f}, Loss: {policy_loss.item():.4f}")
        else:
            print(f"Iter {iteration}/{num_iterations}, "
                  f"Policy Loss: {policy_loss.item():.4f}, Success Rate: {sr:.2f}")

    # 保存最终模型
    torch.save(policy.state_dict(), 'policy_rl.pth')
    torch.save(steps_net.state_dict(), 'steps_net_rl.pth')
    print("Stage 2 models saved: policy_rl.pth, steps_net_rl.pth")

    return success_rates


if __name__ == "__main__":
    pass
