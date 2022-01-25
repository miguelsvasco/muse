import torch
import numpy as np
from muse.evaluation.atari.hyperhot.rl.frame_buffer import FrameBuffer
from muse.evaluation.atari.hyperhot.rl.joint_frame_buffer import JointFrameBuffer


def _discount_rewards(rewards, gamma):
    discounted_reward = rewards[-1]
    for t in reversed(range(len(rewards) - 1)):
        discounted_reward = rewards[t] + gamma * discounted_reward

    return discounted_reward


class Policy(object):
    def __init__(self, policy_net, action_space):
        self.policy_net = policy_net
        self.action_space = action_space

    def select_action(self, state):
        with torch.no_grad():
            action = self.policy_net(state).max(1)[1].detach().cpu().numpy()

        return action


class DQNEvaluator(object):
    def __init__(self, dqn, env, obs_mod, frames_per_state, cuda,
                 preprocess_observation_cb, postprocess_observation_cb):
        assert hasattr(preprocess_observation_cb, '__call__')
        assert hasattr(postprocess_observation_cb, '__call__')

        self.device = 'cuda' if cuda else 'cpu'

        self.dqn = dqn
        self.env = env
        self.obs_mod = obs_mod

        if obs_mod == 'joint':
            self.frame_buffer = JointFrameBuffer(
                frames_per_state,
                preprocessor=preprocess_observation_cb,
                postprocessor=postprocess_observation_cb)
        else:
            self.frame_buffer = FrameBuffer(
                frames_per_state,
                preprocessor=preprocess_observation_cb,
                postprocessor=postprocess_observation_cb)

        self.policy = Policy(self.dqn.net, env.action_space)

    def eval(self, max_episodes, gamma, post_episode_cb=None):
        if post_episode_cb is None:
            post_episode_cb = lambda x: None

        self.env.reset(**{'soft_reset': False})  # force a reset
        for episode_number in range(max_episodes):
            print(f'Eval Episode: {episode_number}/{max_episodes}')

            self.frame_buffer.reset()
            observation = self.env.reset()
            self.frame_buffer.append(observation)
            state = self.frame_buffer.get_state()

            episode_rewards = []
            episode_observations = [observation]
            done = False
            while not done:
                action = self.policy.select_action(state)
                next_observation, reward, done, info = self.env.step(action)
                self.frame_buffer.append(next_observation)
                state = self.frame_buffer.get_state()

                episode_rewards.append(reward)
                episode_observations.append(next_observation)

            discounted_rewards = _discount_rewards(episode_rewards, gamma)
            info = {
                'episode_number': episode_number,
                'total_reward': np.sum(episode_rewards),
                'discounted_reward': discounted_rewards,
                'eval_observations': episode_observations,
                'obs_mod': self.obs_mod
            }
            post_episode_cb(info)

            episode_number += 1

            print(f'**** Eval Episode: {episode_number}/{max_episodes}\t'
                  f'Episode total reward: {np.sum(episode_rewards):.3f}\t'
                  f'Episode discounted reward: {discounted_rewards:.3f}')