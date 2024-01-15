import gym
import math
import random
from collections import namedtuple
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from spikingjelly.clock_driven import neuron, functional
import os


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class NonSpikingLIFNode(neuron.LIFNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, dv: torch.Tensor):
        self.neuronal_charge(dv)
        # self.neuronal_fire()
        # self.neuronal_reset()
        return self.v

class DQSN(nn.Module):
    def __init__(self, hidden_num):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, hidden_num),
            neuron.IFNode(),
            nn.Linear(hidden_num, 2),
            NonSpikingLIFNode(tau=2.0)
        )
        self.T = 16
    def forward(self, x):
        for t in range(self.T):
            self.fc(x)
        return self.fc[-1].v

def train(device, root, hidden_num=128, num_episodes=256):
    BATCH_SIZE = 128
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    TARGET_UPDATE = 10
    steps_done = 0
    env = gym.make('CartPole-v0').unwrapped

    policy_net = DQSN(hidden_num).to(device)
    target_net = DQSN(hidden_num).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters())
    memory = ReplayMemory(10000)

    def select_action(state, steps_done):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * steps_done / EPS_DECAY)
        if sample > eps_threshold:
            with torch.no_grad():
                ac = policy_net(state).max(1)[1].view(1, 1)
                functional.reset_net(policy_net)
                return ac
        else:
            return torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long)

    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        functional.reset_net(target_net)
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        optimizer.step()
        functional.reset_net(policy_net)

    max_duration = 0
    max_pt_path = os.path.join(root, f'policy_net_{hidden_num}_max.pt')
    pt_path = os.path.join(root, f'policy_net_{hidden_num}.pt')

    episode_durations = []

    for i_episode in range(num_episodes):
        # Initialize the environment and state
        env.reset()
        state = torch.zeros([1, 4], dtype=torch.float, device=device)
        for t in count():
            action = select_action(state, steps_done)
            steps_done += 1
            next_state, reward, done, _ = env.step(action.item())
            next_state = torch.from_numpy(next_state).float().to(device).unsqueeze(0)
            reward = torch.tensor([reward], device=device)

            if done:
                next_state = None

            memory.push(state, action, next_state, reward)

            state = next_state
            if done and t + 1 > max_duration:
                max_duration = t + 1
                torch.save(policy_net.state_dict(), max_pt_path)
                print(f'max_duration={max_duration}, save models')

            optimize_model()

            if done:
                print(f'i_episode={i_episode}, duration={t + 1}')
                episode_durations.append(t + 1)
                break
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print('complete')
    torch.save(policy_net.state_dict(), pt_path)
    print('state_dict path is', pt_path)
    # i_episode = 492, duration = 169
    # i_episode = 493, duration = 356
    # i_episode = 494, duration = 296
    # i_episode = 495, duration = 301
    # i_episode = 496, duration = 287
    # i_episode = 497, duration = 303
    # i_episode = 498, duration = 285
    # i_episode = 499, duration = 329
    # i_episode = 500, duration = 303
    # i_episode = 501, duration = 389
    # i_episode = 502, duration = 544
    # i_episode = 503, duration = 619
    # i_episode = 504, duration = 443
    # i_episode = 505, duration = 441
    # i_episode = 506, duration = 322
    # i_episode = 507, duration = 575
    # i_episode = 508, duration = 384
    # i_episode = 509, duration = 715
    # i_episode = 510, duration = 3051
    # i_episode = 511, duration = 571
    # complete
    # state_dict path is./ policy_net_256.pt

def play(device, pt_path, hidden_num, played_frames=60, save_fig_num=0, fig_dir=None, figsize=(12, 6), firing_rates_plot_type='bar', heatmap_shape=None):
    import numpy as np
    from matplotlib import pyplot as plt
    import matplotlib.ticker
    plt.rcParams['figure.figsize'] = figsize
    # plt.rcParams['figure.dpi'] = 200
    plt.ion()
    env = gym.make('CartPole-v0').unwrapped

    policy_net = DQSN(hidden_num).to(device)
    policy_net.load_state_dict(torch.load(pt_path, map_location=device))

    env.reset()
    state = torch.zeros([1, 4], dtype=torch.float, device=device)

    with torch.no_grad():
        functional.set_monitor(policy_net, True)
        delta_lim = 0
        over_score = 1e9
        for i in count():
            # plt.clf()
            LIF_v = policy_net(state)  # shape=[1, 2]
            action = LIF_v.max(1)[1].view(1, 1).item()
            if firing_rates_plot_type == 'bar':
                plt.subplot2grid((2, 9), (1, 0), colspan=3)
            elif firing_rates_plot_type == 'heatmap':
                plt.subplot2grid((2, 3), (1, 0))
            plt.xticks(np.arange(2), ('Left', 'Right'))
            plt.ylabel('Voltage')
            plt.title('Voltage of LIF neurons at last time step')
            delta_lim = (LIF_v.max() - LIF_v.min()) * 0.5
            plt.ylim(LIF_v.min() - delta_lim, LIF_v.max() + delta_lim)
            plt.yticks([])
            plt.text(0, LIF_v[0][0], str(round(LIF_v[0][0].item(), 2)), ha='center')
            plt.text(1, LIF_v[0][1], str(round(LIF_v[0][1].item(), 2)), ha='center')

            plt.bar(np.arange(2), LIF_v.squeeze(), color=['r', 'gray'] if action == 0 else ['gray', 'r'], width=0.5)
            if LIF_v.min() - delta_lim < 0:
                plt.axhline(0, color='black', linewidth=0.1)

            IF_spikes = np.asarray(policy_net.fc[1].monitor['s'])  # shape=[16, 1, 256]
            firing_rates = IF_spikes.mean(axis=0).squeeze()
            if firing_rates_plot_type == 'bar':
                plt.subplot2grid((2, 9), (0, 4), rowspan=2, colspan=5)
            elif firing_rates_plot_type == 'heatmap':
                plt.subplot2grid((2, 3), (0, 1), rowspan=2, colspan=2)
            plt.title('Firing rates of IF neurons')

            if firing_rates_plot_type == 'bar':
                # 绘制柱状图
                plt.xlabel('Neuron index')
                plt.ylabel('Firing rate')
                plt.xlim(0, firing_rates.size)
                plt.ylim(0, 1.01)
                plt.bar(np.arange(firing_rates.size), firing_rates, width=0.5)

            elif firing_rates_plot_type == 'heatmap':
                # 绘制热力图
                heatmap = plt.imshow(firing_rates.reshape(heatmap_shape), vmin=0, vmax=1, cmap='ocean')
                plt.gca().invert_yaxis()
                cbar = heatmap.figure.colorbar(heatmap)
                cbar.ax.set_ylabel('Magnitude', rotation=90, va='top')
            functional.reset_net(policy_net)
            subtitle = f'Position={state[0][0].item(): .2f}, Velocity={state[0][1].item(): .2f}, Pole Angle={state[0][2].item(): .2f}, Pole Velocity At Tip={state[0][3].item(): .2f}, Score={i}'

            state, reward, done, _ = env.step(action)
            if done:
                over_score = min(over_score, i)
                subtitle = f'Game over, Score={over_score}'
            plt.suptitle(subtitle)

            state = torch.from_numpy(state).float().to(device).unsqueeze(0)
            screen = env.render(mode='rgb_array').copy()
            screen[300, :, :] = 0  # 画出黑线
            if firing_rates_plot_type == 'bar':
                plt.subplot2grid((2, 9), (0, 0), colspan=3)
            elif firing_rates_plot_type == 'heatmap':
                plt.subplot2grid((2, 3), (0, 0))
            plt.xticks([])
            plt.yticks([])
            plt.title('Game screen')
            plt.imshow(screen, interpolation='bicubic')
            plt.pause(0.001)
            if i < save_fig_num:
                plt.savefig(os.path.join(fig_dir, f'{i}.png'))
            if done and i >= played_frames:
                env.close()
                plt.close()
                break