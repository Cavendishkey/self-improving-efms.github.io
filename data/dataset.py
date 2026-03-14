from typing import Any, NamedTuple
import numpy as np

NestedArray = Any

class DataTuple(NamedTuple):
    observation: NestedArray
    action: NestedArray
    time_to_success: NestedArray
    reward: NestedArray
    discount: NestedArray
    next_observation: NestedArray


def generate_dataset(env, pd_controller, num_episodes=10000, num_waypoints_per_episode=5,
                     episode_len_discard_thresh=10):
    episodes = []

    while len(episodes) < num_episodes:
        traj = []
        ts = env.reset()
        cur_obs = ts.observation
        succ = env.success()

        waypoint_idx = 0
        if num_waypoints_per_episode == 0:
            cur_waypoint = cur_obs['goal_pos']
        else:
            cur_waypoint = env.sample_goal()
        waypoint_succ = env.success(waypoint=cur_waypoint)

        while not succ:
            if waypoint_succ:
                waypoint_idx += 1
                waypoint_idx = min(waypoint_idx, num_waypoints_per_episode)
                if waypoint_idx == num_waypoints_per_episode:
                    cur_waypoint = cur_obs['goal_pos']
                else:
                    cur_waypoint = env.sample_goal()

            act = pd_controller(cur_obs['cur_pos'], cur_obs['cur_vel'], cur_waypoint)
            ts = env.step(act)
            next_obs = ts.observation

            traj.append({
                'observation': cur_obs,
                'action': act,
                'reward': ts.reward if ts.reward is not None else 0.0,
                'discount': 1.0,
                'next_observation': next_obs,
            })

            cur_obs = next_obs
            succ = env.success()
            waypoint_succ = env.success(waypoint=cur_waypoint)

        # 添加最后一步（discount=0）
        act = pd_controller(cur_obs['cur_pos'], cur_obs['cur_vel'], cur_waypoint)
        traj.append({
            'observation': cur_obs,
            'action': act,
            'reward': ts.reward if ts.reward is not None else 0.0,
            'discount': 0.0,
            'next_observation': cur_obs,
        })

        # 丢弃过短的 episode
        traj_len = len(traj)
        if traj_len < episode_len_discard_thresh:
            continue

        # 将列表转换为 numpy 数组堆叠
        # 注意：observation 和 next_observation 是字典，需要先展平成向量
        obs_list = []
        act_list = []
        reward_list = []
        discount_list = []
        next_obs_list = []

        for step in traj:
            obs_vec = np.concatenate([step['observation']['cur_pos'],
                                      step['observation']['cur_vel'],
                                      step['observation']['goal_pos']])
            next_obs_vec = np.concatenate([step['next_observation']['cur_pos'],
                                           step['next_observation']['cur_vel'],
                                           step['next_observation']['goal_pos']])
            obs_list.append(obs_vec)
            act_list.append(step['action'])
            reward_list.append(step['reward'])
            discount_list.append(step['discount'])
            next_obs_list.append(next_obs_vec)

        # 堆叠成数组
        obs_arr = np.stack(obs_list, axis=0).astype(np.float32)
        act_arr = np.stack(act_list, axis=0).astype(np.float32)
        reward_arr = np.array(reward_list, dtype=np.float32)
        discount_arr = np.array(discount_list, dtype=np.float32)
        next_obs_arr = np.stack(next_obs_list, axis=0).astype(np.float32)

        # 计算 time_to_success（从后往前递减）
        time_to_success_arr = np.arange(traj_len - 1, -1, -1, dtype=np.float32)

        # 保存为 DataTuple
        ep_tuple = DataTuple(
            observation=obs_arr,
            action=act_arr,
            time_to_success=time_to_success_arr,
            reward=reward_arr,
            discount=discount_arr,
            next_observation=next_obs_arr,
        )
        episodes.append(ep_tuple)

    # 将所有 episodes 拼接成一个大数组
    all_obs = np.concatenate([ep.observation for ep in episodes], axis=0)
    all_act = np.concatenate([ep.action for ep in episodes], axis=0)
    all_time = np.concatenate([ep.time_to_success for ep in episodes], axis=0)
    all_reward = np.concatenate([ep.reward for ep in episodes], axis=0)
    all_discount = np.concatenate([ep.discount for ep in episodes], axis=0)
    all_next_obs = np.concatenate([ep.next_observation for ep in episodes], axis=0)

    all_tuples = DataTuple(
        observation=all_obs,
        action=all_act,
        time_to_success=all_time,
        reward=all_reward,
        discount=all_discount,
        next_observation=all_next_obs,
    )

    # 打印统计信息
    print(f"Num Episodes: {len(episodes)}")
    lens = [ep.observation.shape[0] for ep in episodes]
    print(f"Episode Lens: mean={np.mean(lens):.2f}, std={np.std(lens):.2f}")
    print(f"Max Episode Len: {np.max(lens)}")
    print(f"Min Episode Len: {np.min(lens)}")

    return episodes, all_tuples
