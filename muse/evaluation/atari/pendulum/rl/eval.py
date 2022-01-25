import torch
import numpy as np
from muse.evaluation.atari.pendulum.rl.frame_buffer import FrameBuffer
from muse.evaluation.atari.pendulum.rl.joint_frame_buffer import JointFrameBuffer


class Policy(object):
    def __init__(self, actor, action_space):
        self.actor = actor
        self.action_space = action_space

    def select_action(self, state):
        with torch.no_grad():
            action = self.actor(state).squeeze(0).detach().cpu().numpy()

        action = np.clip(action, self.action_space.low, self.action_space.high)
        return action


class DDPGEvaluator(object):
    def __init__(self, ddpg, env, obs_mod, frames_per_state, cuda,
                 preprocess_observation_cb, postprocess_observation_cb):
        assert hasattr(preprocess_observation_cb, '__call__')
        assert hasattr(postprocess_observation_cb, '__call__')

        if cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.ddpg = ddpg
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



        self.policy = Policy(self.ddpg.actor, env.action_space)

    def eval(self, max_episodes, max_episode_length, post_episode_cb=None):
        if post_episode_cb is None:
            post_episode_cb = lambda x: None

        for episode_number in range(max_episodes):
            print(f'Eval Episode: {episode_number}/{max_episodes}')

            self.frame_buffer.reset()
            observation = self.env.reset()
            self.frame_buffer.append(observation)
            state = self.frame_buffer.get_state()

            episode_reward_sum = 0.0
            episode_observations = [observation]
            for episode_frame_number in range(max_episode_length):
                action = self.policy.select_action(state)
                action = torch.from_numpy(action).to(self.device)

                next_observation, reward, done, info = self.env.step(
                    action.cpu().numpy())

                episode_reward_sum += reward

                reward = torch.tensor([reward], device=self.device)

                self.frame_buffer.append(next_observation)
                next_state = self.frame_buffer.get_state()
                state = next_state

                episode_observations.append(next_observation)
                if done:
                    break

            avg_reward = episode_reward_sum / (episode_frame_number + 1)
            info = {
                'episode_number': episode_number,
                'avg_reward': avg_reward,
                'eval_observations': episode_observations,
                'obs_mod':self.obs_mod
            }
            post_episode_cb(info)

            episode_number += 1

            print(f'**** Eval Episode: {episode_number}/{max_episodes}\t'
                  f'Episode avg reward: {avg_reward:.3f}')
