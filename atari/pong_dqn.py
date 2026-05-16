import time
import typing as tt
from collections import deque
from dataclasses import dataclass

import ale_py
import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim
from torch.utils.tensorboard.writer import SummaryWriter

State = np.ndarray
Action = int
BatchTensors = tt.Tuple[
    torch.ByteTensor,  # state
    torch.LongTensor,  # action
    torch.Tensor,  # reward
    torch.BoolTensor,  # done
    torch.ByteTensor,  # next state
]


@dataclass
class Experience:
    state: State
    action: Action
    reward: float
    done: bool
    next_state: State


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        size = self.conv(torch.zeros(1, *input_shape)).size()[-1]
        self.fc = nn.Sequential(
            nn.Linear(size, 512), nn.ReLU(), nn.Linear(512, n_actions)
        )

    def forward(self, x: torch.ByteTensor):
        # scale on GPU
        xx = x / 255.0
        return self.fc(self.conv(xx))


class ExperienceBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def put(self, exp: Experience):
        self.buffer.append(exp)

    def sample_batch(self, batch_size: int):
        indicies = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[idx] for idx in indicies]

    def __len__(self) -> int:
        return len(self.buffer)


class Agent:
    def __init__(self, env: gym.Env, exp_buffer: ExperienceBuffer) -> None:
        self.total_reward = 0.0
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state, _ = self.env.reset()
        self.total_reward = 0.0

    @torch.no_grad
    def play_step(self, net: DQN, device: torch.device, epsilon=0.0):
        done_reward = None

        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_t = torch.as_tensor(self.state, device=device)
            state_t.unsqueeze_(0)
            qvals_t = net(state_t)
            action = torch.argmax(qvals_t, dim=1).item()

        next_state, reward, terminated, truncated, _ = self.env.step(action)
        is_done = terminated or truncated

        self.total_reward += float(reward)
        self.exp_buffer.put(
            Experience(self.state, int(action), float(reward), is_done, next_state)
        )
        self.state = next_state

        if is_done:
            done_reward = self.total_reward
            self._reset()

        return done_reward


def batch_to_tensors(batch: tt.List[Experience], device: torch.device) -> BatchTensors:
    states, actions, rewards, dones, new_state = [], [], [], [], []
    for e in batch:
        states.append(e.state)
        actions.append(e.action)
        rewards.append(e.reward)
        dones.append(e.done)
        new_state.append(e.next_state)

    states_t = torch.ByteTensor(np.asarray(states), device=device)
    actions_t = torch.LongTensor(actions, device=device)
    rewards_t = torch.FloatTensor(rewards, device=device)
    dones_t = torch.BoolTensor(dones, device=device)
    new_states_t = torch.ByteTensor(np.asarray(new_state), device=device)

    return states_t, actions_t, rewards_t, dones_t, new_states_t


def calc_loss(
    batch: tt.List[Experience],
    net: DQN,
    tgt_net: DQN,
    device: torch.device,
) -> torch.Tensor:
    states_t, actions_t, rewards_t, dones_t, new_states_t = batch_to_tensors(
        batch, device
    )

    state_action_values = net(states_t).gather(1, actions_t.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_state_values = tgt_net(new_states_t).max(1)[0]
        next_state_values[dones_t] = 0.0
        next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_t
    return nn.MSELoss()(state_action_values, expected_state_action_values)


MEAN_REWARD_BOUND = 19

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01


def make_env():
    gym.register_envs(ale_py)

    env = gym.make("PongNoFrameskip-v4", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        episode_trigger=lambda num: num % 50 == 0,
        video_folder="saved-video-folder",
        name_prefix="pong",
    )
    env = gym.wrappers.AtariPreprocessing(env, scale_obs=True)
    env = gym.wrappers.FrameStackObservation(env, 4)

    return env


if __name__ == "__main__":
    env = make_env()

    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")
    device = torch.device(device)

    writer = SummaryWriter()
    net = DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = DQN(env.observation_space.shape, env.action_space.n).to(device)

    exp_buffer = ExperienceBuffer(REPLAY_SIZE)
    epsilon = EPSILON_START
    agent = Agent(env, exp_buffer)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_m_reward = None

    while True:
        frame_idx += 1
        epsilon = max(
            EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME
        )

        reward = agent.play_step(net, device, epsilon)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            m_reward = np.mean(total_rewards[-100:])
            print(
                f"frame {frame_idx}: done {len(total_rewards)} games, reward {m_reward:.3f}, "
                f"eps {epsilon:.2f}, speed {speed:.2f} f/s"
            )
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", m_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            if best_m_reward is None or best_m_reward < m_reward:
                torch.save(
                    net.state_dict(), "saved-dqn-models/pong-best_%.0f.dat" % m_reward
                )
                if best_m_reward is not None:
                    print(f"Best reward updated {best_m_reward:.3f} -> {m_reward:.3f}")
                best_m_reward = m_reward
            if m_reward > MEAN_REWARD_BOUND:
                print("Solved in %d frames!" % frame_idx)
                break
        if len(exp_buffer) < REPLAY_START_SIZE:
            continue
        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = exp_buffer.sample_batch(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device)
        loss_t.backward()
        optimizer.step()
    writer.close()
