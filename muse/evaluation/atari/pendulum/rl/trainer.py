import torch
import random
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
from muse.evaluation.atari.pendulum.rl.joint_frame_buffer import JointFrameBuffer

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'terminal'))


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) +
                                param.data * tau)


def _discount_rewards(rewards, gamma):
    discounted_reward = rewards[-1]
    for t in reversed(range(len(rewards) - 1)):
        discounted_reward = rewards[t] + gamma * discounted_reward

    return discounted_reward


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


class OUNoise(object):
    def __init__(self,
                 action_space,
                 mu=0.0,
                 theta=0.15,
                 max_sigma=0.3,
                 min_sigma=0.3,
                 decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(
            self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(
            1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


class Policy(object):
    def __init__(self,
                 actor,
                 action_space,
                 replay_buffer_size,
                 ou_mu=0.0,
                 ou_theta=0.15,
                 ou_max_sigma=0.3,
                 ou_min_sigma=0.0,
                 ou_decay_period=1000000):
        self.actor = actor
        self.action_space = action_space
        self.replay_buffer_size = replay_buffer_size
        self.random_process = OUNoise(
            action_space=action_space,
            mu=ou_mu,
            theta=ou_theta,
            max_sigma=ou_max_sigma,
            min_sigma=ou_min_sigma,
            decay_period=ou_decay_period)

    def random_action(self):
        return self.action_space.sample()

    def select_action(self, state, frame_number, evaluation=False):
        if (evaluation) or (frame_number >= self.replay_buffer_size):
            with torch.no_grad():
                action = self.actor(state).squeeze(0).detach().cpu().numpy()

            action += (not evaluation) * self.random_process.get_action(
                action, frame_number).squeeze(0)
            action = np.clip(action, self.action_space.low,
                             self.action_space.high)

        else:
            action = self.random_action()

        return action


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

    def stats(self):
        unrolled = Transition(*zip(*self.memory))
        reward = torch.cat(unrolled.reward).cpu().numpy()

        return np.mean(reward), np.std(reward)

    def __len__(self):
        return len(self.memory)


class DDPGTrainer(object):
    def __init__(self, ddpg, env, frames_per_state, gamma, actor_learning_rate,
                 critic_learning_rate, tau, batch_size, memory_size,
                 random_process_config, cuda, preprocess_observation_cb,
                 postprocess_observation_cb):
        assert hasattr(preprocess_observation_cb, '__call__')
        assert hasattr(postprocess_observation_cb, '__call__')

        if cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.ddpg = ddpg
        self.env = env

        self.actor_optim = optim.Adam(
            self.ddpg.actor.parameters(), lr=actor_learning_rate)
        self.critic_optim = optim.Adam(
            self.ddpg.critic.parameters(), lr=critic_learning_rate)

        self.memory_size = memory_size
        self.memory = ReplayMemory(memory_size)
        self.frame_buffer = JointFrameBuffer(
            frames_per_state,
            preprocessor=preprocess_observation_cb,
            postprocessor=postprocess_observation_cb)

        self.policy = Policy(
            self.ddpg.actor,
            env.action_space,
            memory_size,
            ou_mu=random_process_config['ou_mu'],
            ou_theta=random_process_config['ou_theta'],
            ou_max_sigma=random_process_config['ou_max_sigma'],
            ou_min_sigma=random_process_config['ou_min_sigma'],
            ou_decay_period=random_process_config['ou_decay_period'])

        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau

    def eval(self, frame_number, episode_number, n_eval_episodes,
             post_eval_cb):
        print(f'**** Eval Episode: {episode_number}')

        self.ddpg.eval()

        observations = []
        total_rewards = []
        for episode in range(n_eval_episodes):
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
                episode_observations.append(next_observation)

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

        self.ddpg.train()

        frame_number = 0
        episode_number = 0
        avg_episode_total_reward = FixedHorizonAverageMeter(50)
        avg_rewards = FixedHorizonAverageMeter(1000)
        avg_critic_losses = FixedHorizonAverageMeter(1000)
        avg_actor_losses = FixedHorizonAverageMeter(1000)

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

            replay_memory_filled = frame_number > self.memory_size
            if replay_memory_filled:
                critic_loss, actor_loss = self.optimize_model()
                avg_critic_losses.update(critic_loss)
                avg_actor_losses.update(actor_loss)

            # log every 500 frames
            should_log = frame_number % 500 == 0
            if should_log and replay_memory_filled:
                print(
                    f'===> Train Episode: {episode_number} - {frame_number}/{max_frames}\t'
                    f'Episode avg critic loss: {avg_critic_losses.avg:.3f}\t'
                    f'Episode avg actor loss: {avg_actor_losses.avg:.3f}\t'
                    f'Episode avg episode total reward: {avg_episode_total_reward.avg:.3f}\t'
                    f'Episode avg reward: {avg_rewards.avg:.3f}\t'
                    f'ReplayBuf avg reward: {self.memory.stats()[0]:.3f}')
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
                    'avg_critic_loss': avg_critic_losses.avg,
                    'avg_actor_loss': avg_actor_losses.avg,
                    'avg_reward': avg_rewards.avg,
                    'avg_episode_total_reward': avg_episode_total_reward.avg,
                    'last_episode_total_reward': total_episode_reward,
                    'replay_buf_avg_reward': self.memory.stats()[0]
                }
                post_episode_cb(info)
                episode_number += 1
                new_episode = True

                should_eval = (episode_number % eval_frequency == 0)
                if should_eval:
                    self.eval(frame_number, episode_number, eval_length,
                              post_eval_cb)
                    self.ddpg.train()

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

        # Update critic
        next_target_actions = self.ddpg.actor_target(next_state_batch)
        target_next_state_action_values = self.ddpg.critic_target(
            [next_state_batch, next_target_actions]).detach()
        target_state_action_values = reward_batch + self.gamma * (
            1.0 - terminal_batch) * target_next_state_action_values
        state_action_values = self.ddpg.critic([state_batch, action_batch])

        critic_loss = F.mse_loss(
            state_action_values.double(), target_state_action_values.double(), reduction='mean')
        self.ddpg.critic.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1)
        self.critic_optim.step()

        # Update actor
        actor_loss = -self.ddpg.critic(
            [state_batch, self.ddpg.actor(state_batch)]).mean()
        self.ddpg.actor.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1)
        self.actor_optim.step()

        soft_update(self.ddpg.actor_target, self.ddpg.actor, self.tau)
        soft_update(self.ddpg.critic_target, self.ddpg.critic, self.tau)

        return critic_loss.cpu().data.numpy(), actor_loss.cpu().data.numpy()