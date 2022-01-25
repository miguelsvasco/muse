import torch
import random
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
from muse.evaluation.atari.hyperhot.rl.joint_frame_buffer import JointFrameBuffer

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'terminal'))


def _discount_rewards(rewards, gamma):
    discounted_reward = rewards[-1]
    for t in reversed(range(len(rewards) - 1)):
        discounted_reward = rewards[t] + gamma * discounted_reward

    return discounted_reward


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self._running_reward_sum = 0.0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        transition = Transition(*args)
        self._update_running_reward_sum(transition.reward.cpu().item())

        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def _update_running_reward_sum(self, new_reward):
        self._running_reward_sum += new_reward
        if self.memory[self.position] is not None:
            old_reward = self.memory[self.position].reward.cpu().item()
            self._running_reward_sum -= old_reward

    def avg_reward(self):
        return self._running_reward_sum / len(self.memory)

    def __len__(self):
        return len(self.memory)


class FixedHorizonAverageMeter(object):
    def __init__(self, horizon):
        self.horizon = horizon
        self.reset()

    def reset(self):
        self.vals = []
        self.position = 0
        self.avg = 0

    def update(self, val):
        if len(self.vals) < self.horizon:
            self.vals.append(None)

        self.vals[self.position] = val
        self.position = (self.position + 1) % self.horizon
        self.avg = np.mean(self.vals)


class EpsGreedyPolicy(object):
    def __init__(self,
                 policy_network,
                 action_space,
                 eps_initial=1,
                 eps_end=0.01,
                 n_annealing_frames=1000000,
                 replay_buffer_start_size=50000,
                 eps_evaluation=0.0):
        self.policy_network = policy_network
        self.action_space = action_space
        self.eps_initial = eps_initial
        self.eps_end = eps_end
        self.n_annealing_frames = n_annealing_frames
        self.replay_buffer_start_size = replay_buffer_start_size
        self.eps_evaluation = eps_evaluation

    def select_action(self, state, frame_number, evaluation=False):
        if evaluation:
            eps = self.eps_evaluation
        elif frame_number < self.replay_buffer_start_size:
            eps = 1.0
        else:
            eps = self.eps_initial + (self.eps_end - self.eps_initial) * (
                frame_number -
                self.replay_buffer_start_size) / self.n_annealing_frames
            eps = max(eps, self.eps_end)

        if np.random.rand(1) > eps:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected
                # reward.
                return self.policy_network(state).max(
                    1)[1].detach().cpu().numpy()
        else:
            return np.array([self.action_space.sample()])


class DQNTrainer(object):
    def __init__(self, dqn, env, frames_per_state, gamma, learning_rate,
                 batch_size, memory_size, policy_network_update_freq,
                 target_network_update_freq, eps_initial, eps_end, eps_decay,
                 replay_buffer_start_size, cuda, preprocess_observation_cb,
                 postprocess_observation_cb, eval_process_observation_cb):
        assert hasattr(preprocess_observation_cb, '__call__')
        assert hasattr(postprocess_observation_cb, '__call__')
        assert hasattr(eval_process_observation_cb, '__call__')

        self.device = 'cuda' if cuda else 'cpu'

        self.dqn = dqn
        self.env = env

        self.optim = optim.Adam(self.dqn.net.parameters(), lr=learning_rate)

        self.memory_size = memory_size
        self.replay_buffer_start_size = replay_buffer_start_size
        self.memory = ReplayMemory(memory_size)
        self.frame_buffer = JointFrameBuffer(
            frames_per_state,
            preprocessor=preprocess_observation_cb,
            postprocessor=postprocess_observation_cb)

        self.policy = EpsGreedyPolicy(
            self.dqn.net,
            env.action_space,
            eps_initial=eps_initial,
            eps_end=eps_end,
            n_annealing_frames=eps_decay,
            replay_buffer_start_size=replay_buffer_start_size,
            eps_evaluation=0.0)

        self.gamma = gamma
        self.batch_size = batch_size
        self.policy_network_update_freq = policy_network_update_freq
        self.target_network_update_freq = target_network_update_freq

        self.eval_process_observation_cb = eval_process_observation_cb

    def eval(self, frame_number, episode_number, n_eval_episodes,
             post_eval_cb):
        print(f'**** Eval Episode: {episode_number}')

        self.dqn.eval()

        observations = []
        total_rewards = []
        self.env.reset(**{'soft_reset': False})  # force a reset
        for episode in range(n_eval_episodes):
            print(f'=====Eval epoch: {episode}/{n_eval_episodes}')
            self.frame_buffer.reset()
            observation = self.env.reset()
            self.frame_buffer.append(observation)
            state = self.frame_buffer.get_state()

            episode_rewards = []
            episode_observations = []

            done = False
            while not done:
                action = self.policy.select_action(
                    state, frame_number, evaluation=True)

                next_observation, reward, done, info = self.env.step(action)
                self.frame_buffer.append(next_observation)
                state = self.frame_buffer.get_state()

                episode_rewards.append(reward)
                episode_observations.append(
                    self.eval_process_observation_cb(next_observation))

            total_rewards.append(
                _discount_rewards(episode_rewards, self.gamma))
            observations.append(episode_observations)

        info['frame_number'] = frame_number
        info['eval_avg_reward'] = np.mean(total_rewards)
        info['eval_observations'] = observations
        post_eval_cb(info)

    def train(self,
              max_frames,
              eval_frequency,
              eval_length,
              post_episode_cb=None,
              post_eval_cb=None):
        if post_episode_cb is None:
            post_episode_cb = lambda x: None
        if post_eval_cb is None:
            post_eval_cb = lambda x: None

        self.dqn.train()

        frame_number = 0
        episode_number = 0
        avg_episode_total_reward = FixedHorizonAverageMeter(50)
        avg_rewards = FixedHorizonAverageMeter(1000)
        avg_losses = FixedHorizonAverageMeter(1000)

        new_episode = True
        while frame_number < max_frames:
            if new_episode:
                self.frame_buffer.reset()
                observation = self.env.reset()
                self.frame_buffer.append(observation)
                state = self.frame_buffer.get_state()

                print(
                    f'Train Episode: {episode_number} - {frame_number}/{max_frames}'
                )

                episode_rewards = []

                new_episode = False

            action = self.policy.select_action(state, frame_number)
            next_observation, reward, done, info = self.env.step(action)

            self.frame_buffer.append(next_observation)
            next_state = self.frame_buffer.get_state()
            torch_action, torch_reward = (torch.from_numpy(action).to(
                self.device), torch.tensor([reward], device=self.device))
            self.memory.push(state, torch_action, next_state, torch_reward,
                             done)
            state = next_state

            replay_memory_filled = frame_number > self.replay_buffer_start_size
            update_policy_network = (
                frame_number % self.policy_network_update_freq) == 0
            update_target_network = (
                frame_number % self.target_network_update_freq) == 0
            if update_policy_network and replay_memory_filled:
                loss = self.optimize_model()
                avg_losses.update(loss)
            if update_target_network:
                self.dqn.target.load_state_dict(self.dqn.net.state_dict())

            # log every 500 frames
            should_log = frame_number % 500 == 0
            if should_log and replay_memory_filled:
                print(
                    f'===> Train Episode: {episode_number} - {frame_number}/{max_frames}\t'
                    f'Episode avg loss: {avg_losses.avg:.3f}\t'
                    f'Episode avg reward: {avg_episode_total_reward.avg:.3f}\t'
                    f'ReplayBuf avg reward: {self.memory.avg_reward():.3f}')
            elif should_log and (not replay_memory_filled):
                print(f'Fill ReplayMemory: {frame_number}/{self.memory_size}')

            avg_rewards.update(reward)
            episode_rewards.append(reward)
            if done:
                total_episode_reward = _discount_rewards(
                    episode_rewards, self.gamma)
                avg_episode_total_reward.update(total_episode_reward)
                info = {
                    'frame_number': frame_number,
                    'avg_loss': avg_losses.avg,
                    'avg_reward': avg_rewards.avg,
                    'avg_episode_total_reward': avg_episode_total_reward.avg,
                    'last_episode_total_reward': total_episode_reward,
                    'replay_buf_avg_reward': self.memory.avg_reward()
                }
                post_episode_cb(info)
                episode_number += 1
                new_episode = True

                should_eval = (episode_number % eval_frequency == 0)
                if should_eval:
                    self.eval(frame_number, episode_number, eval_length,
                              post_eval_cb)
                    self.dqn.train()

            frame_number += 1

        self.eval(frame_number, episode_number, eval_length, post_eval_cb)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return 0.0, 0.0

        transitions = self.memory.sample(self.batch_size)

        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        next_state_batch = torch.cat(batch.next_state)
        action_batch = torch.cat(batch.action).unsqueeze(1)
        reward_batch = torch.cat(batch.reward).unsqueeze(1)
        terminal_batch = torch.tensor(
            batch.terminal, dtype=torch.float).to(self.device).unsqueeze(1)

        # Compute Q(s_t, a) - the model computes Q(s_t, :), then we
        # select the columns of actions taken. These are the actions
        # which would've been taken for each batch state according to
        # policy network
        state_action_values = self.dqn.net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states. Expected values of
        # actions for non_final_next_states are computed based on the
        # "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either
        # the expected state value or 0 in case the state was final.

        # DQN:
        # next_state_values = self.dqn.target(next_state_batch).max(
        #     1)[0].view(-1, 1).detach()

        # DDQN
        _, next_state_actions = self.dqn.net(next_state_batch).max(
            1, keepdim=True)
        next_state_values = self.dqn.target(next_state_batch).gather(
            1, next_state_actions).detach()

        target_state_action_values = reward_batch + self.gamma * (
            1.0 - terminal_batch) * next_state_values

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values,
                                target_state_action_values)
        self.optim.zero_grad()
        loss.backward()
        for param in self.dqn.net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optim.step()

        return loss.detach().cpu().numpy()