#!/usr/bin/env python3
import typing as tt

import gymnasium as gym
import numpy as np
import ptan
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym import Env
from ptan.experience import ExperienceSourceFirstLast
from torch.utils.tensorboard.writer import SummaryWriter

GAMMA = 0.99
LEARNING_RATE = 0.01
N_EPISODES = 4
MAX_EPISODES = 1000


class PGN(nn.Module):
    def __init__(self, input_size: int, n_actions: int):
        super(PGN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        episode_trigger=lambda num: num % 50 == 0,
        video_folder="saved-video-folder",
        name_prefix="cartpole-",
    )

    writer = SummaryWriter(comment="-cartpole-reinforce")

    net = PGN(env.observation_space.shape[0], env.action_space.n)
    print(net)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    def calc_qvals(rewards):
        sum = 0.0
        res = []
        for r in reversed(rewards):
            sum *= GAMMA
            sum += r
            res.append(sum)

        return list(reversed(res))

    def play_episodes():

        states = []
        actions = []
        rewards = []
        qvals = []
        episode_reward = []

        for _ in range(N_EPISODES):
            state, info = env.reset()
            is_done = False
            cur_rewards = []
            while not is_done:
                logits = net.forward(torch.as_tensor(np.asarray(state)))
                probs = F.softmax(logits, -1).detach().numpy().astype(np.float64)
                probs = probs / probs.sum()

                action = env.action_space.sample(probability=probs)
                state, reward, truncated, terminated, info = env.step(action)

                states.append(state)
                actions.append(action)
                cur_rewards.append(reward)

                is_done = truncated or terminated

            rewards.extend(cur_rewards)
            qvals.extend(calc_qvals(cur_rewards))
            episode_reward.append(np.sum(cur_rewards))

        return states, actions, rewards, qvals, episode_reward

    step_idx = 0
    total_rewards = []
    converged = False
    while not converged and step_idx < MAX_EPISODES:
        states, actions, rewards, qvals, episode_rewards = play_episodes()

        for reward in episode_rewards:
            total_rewards.append(reward)
            mean_rewards = float(np.mean(total_rewards[-100:]))
            print(
                f"episode {step_idx}: reward: {reward:6.2f}, mean_100: {mean_rewards:6.2f}"
            )
            writer.add_scalar("reward", reward, step_idx)
            writer.add_scalar("reward_100", mean_rewards, step_idx)
            step_idx += 1
            if mean_rewards > 450:
                print(f"solved in {step_idx} episodes!")
                converged = True

        # Normalize qvals
        qvals = np.array(qvals, dtype=np.float32)
        qvals = (qvals - qvals.mean()) / (qvals.std() + 1e-8)

        states_t = torch.as_tensor(np.asarray(states))
        actions_t = torch.as_tensor(np.asarray(actions))
        qvals_t = torch.as_tensor(np.asarray(qvals))

        optimizer.zero_grad()
        logits_t = net(states_t)
        log_probs_t = F.log_softmax(logits_t, dim=1)
        batch_idx = range(len(states))
        act_probs_t = log_probs_t[batch_idx, actions_t]
        log_prob_qvals_t = act_probs_t * qvals_t
        loss_t = -log_prob_qvals_t.mean()

        loss_t.backward()
        optimizer.step()

    writer.close()
