import os
import sacred
import imageio
import numpy as np
from muse.evaluation.atari.hyperhot.rl.model import DQN
import muse.scenarios.multimodal_atari.hyperhot_env as hh
import muse.evaluation.atari.hyperhot.rl.utils as dqn_utils
import muse.evaluation.atari.hyperhot.vae.utils as vae_utils
from muse.evaluation.atari.hyperhot.rl.trainer import DQNTrainer
import muse.evaluation.atari.hyperhot.ingredients as ingredients
from muse.scenarios.multimodal_atari.hyperhot_env import HyperhotEnv
from muse.evaluation.atari.hyperhot.hyperhot_processor import Processor

ex = sacred.Experiment(
    'hyperhot_muse_dqn_train_dqn',
    ingredients=[
        ingredients.hyperhot_ingredient, ingredients.dqn_ingredient,
        ingredients.dqn_eval_ingredient, ingredients.vae_ingredient,
        ingredients.dqn_setup_vae_dep_ingredient,
        ingredients.gpu_ingredient
    ])


@ex.capture
def log_dir_path(folder, _config):
    return os.path.join('./results', f'log_{_config["seed"]}', folder)


@ex.capture
def post_episode_cb(info, _run):
    frame_number = info['frame_number']

    _run.log_scalar('avg_loss', info['avg_loss'], step=frame_number)
    _run.log_scalar('avg_reward', info['avg_reward'], step=frame_number)
    _run.log_scalar(
        'avg_episode_total_reward',
        info['avg_episode_total_reward'],
        step=frame_number)
    _run.log_scalar(
        'last_episode_total_reward',
        info['last_episode_total_reward'],
        step=frame_number)
    _run.log_scalar(
        'replay_buf_avg_reward',
        info['replay_buf_avg_reward'],
        step=frame_number)


class PostEvalCb(object):
    def __init__(self, dqn, optimizer):
        self.dqn = dqn
        self.optimizer = optimizer
        self.best_reward = float('-inf')

    @ex.capture
    def _record_info(self, info, _run):
        frame_number = info['frame_number']
        _run.log_scalar(
            'eval_avg_reward', info['eval_avg_reward'], step=frame_number)

    @ex.capture
    def _record_gif(self, info, _run):
        frame_number = info['frame_number']
        frame_rate = 27  # fps

        # 1st ep, 1st frame, image
        frame_shape = info['eval_observations'][0][0].shape
        black_frames = [
            np.zeros(frame_shape, dtype=np.uint8) for _ in range(frame_rate)
        ]
        white_frames = [
            np.ones(frame_shape, dtype=np.uint8) * 255
            for _ in range(2 * frame_rate)
        ]

        gif_path = os.path.join(
            log_dir_path('results'), f'hyperhot_eval_{frame_number}.gif')
        frames_for_gif = []
        for episode_observations in info['eval_observations']:
            frames_for_gif.extend(episode_observations)
            frames_for_gif.extend(black_frames)
        frames_for_gif.extend(white_frames)

        imageio.mimsave(
            gif_path,
            frames_for_gif,
            duration=1.0 / frame_rate,
            subrectangles=True)
        ex.add_artifact(gif_path)

    @ex.capture
    def _record_checkpoint(self, info, _config):
        avg_reward = info['eval_avg_reward']
        is_best = avg_reward > self.best_reward
        self.best_reward = max(avg_reward, self.best_reward)

        env_config = {
            'n_states': self.dqn.n_states,
            'n_actions': self.dqn.n_actions
        }
        # TODO: Find a better way to do this. _config is a
        # ReadOnlyDict defined by sacred, which is not pickleable. And
        # due to the nested config, we can't simply do dict(_config)
        pickleable_dqn_config = dict(_config['dqn'])
        dqn_utils.save_checkpoint({
            'state_dict': self.dqn.state_dict(),
            'best_reward': self.best_reward,
            'frame_number': info['frame_number'],
            'optimizer': self.optimizer.state_dict(),
            'dqn_config': pickleable_dqn_config,
            'env_config': env_config
        },
                                 is_best,
                                 folder=log_dir_path('trained_models'))

    @ex.capture
    def __call__(self, info, _config, _run):
        self._record_info(info)
        self._record_gif(info)
        self._record_checkpoint(info)


@ex.capture
def train(vae, sound_normalization, _config, _run):
    hyperhot_config = _config['hyperhot']
    dqn_config = _config['dqn']
    dqn_eval_config = _config['dqn_eval']
    vae_dep_config = _config['dqn_setup_vae_dep']

    env = HyperhotEnv(
        num_enemies=hyperhot_config['n_enemies'],
        pacifist_mode=hyperhot_config['pacifist_mode'],
        time_limit=hyperhot_config['time_limit'],
        sound_receivers=[
            hh.SoundReceiver(hh.SoundReceiver.Location[ss])
            for ss in hyperhot_config['sound_receivers']
        ])

    print(f'DQN: states {vae.n_latents} | actions {env.action_space}')
    dqn = DQN(vae.n_latents, env.action_space.n,
              dqn_config['layers_sizes'], _config['gpu']['cuda'])

    processor = Processor(vae, sound_normalization,
                          _config['gpu']['cuda'])
    if dqn_config['condition_on_image']:
        print('DDPG conditioned on image.')
        preprocess = processor.preprocess_just_image
        postprocess = processor.postprocess_just_image
    elif dqn_config['condition_on_sound']:
        print('DDPG conditioned on sound.')
        preprocess = processor.preprocess_just_sound
        postprocess = processor.postprocess_just_sound
    elif dqn_config['condition_on_joint']:
        print('DDPG conditioned on joint.')
        preprocess = processor.preprocess_joint
        postprocess = processor.postprocess_joint

    else:
        raise 'Unknown conditioning option'

    trainer = DQNTrainer(
        dqn,
        env,
        hyperhot_config['n_stack'],
        dqn_config['gamma'],
        dqn_config['learning_rate'],
        dqn_config['batch_size'],
        dqn_config['memory_size'],
        dqn_config['policy_network_update_freq'],
        dqn_config['target_network_update_freq'],
        dqn_config['eps_initial'],
        dqn_config['eps_end'],
        dqn_config['eps_decay'],
        dqn_config['replay_buffer_start_size'],
        _config['gpu']['cuda'],
        preprocess_observation_cb=preprocess,
        postprocess_observation_cb=postprocess,
        eval_process_observation_cb=processor.preprocess_eval_image)

    post_eval_cb = PostEvalCb(dqn, trainer.optim)
    trainer.train(dqn_config['max_frames'], dqn_eval_config['eval_frequency'],
                  dqn_eval_config['eval_length'], post_episode_cb,
                  post_eval_cb)

    ex.add_artifact(
        os.path.join(log_dir_path('trained_models'), 'dqn_checkpoint.pth.tar'),
        name='dqn_last_checkpoint.pth.tar')
    ex.add_artifact(
        os.path.join(log_dir_path('trained_models'), 'best_dqn_model.pth.tar'))


def hyperhot_environment_config_matches(config, rhang_vae_env_config):
    # checks if the config dict is inside the rhang_vae_env_config dict
    return config.items() <= rhang_vae_env_config.items()


def flatten_dict(dd, separator='_', prefix=''):
    return {
        prefix + separator + k if prefix else k: v
        for kk, vv in dd.items()
        for k, v in flatten_dict(vv, separator, kk).items()
    } if isinstance(dd, dict) else {
        prefix: dd
    }


def get_vae_model(config, cuda):
    vae_dep_config = config['dqn_setup_vae_dep']

    filepath = vae_dep_config['file']
    vae, _, vae_env_config = vae_utils.load_checkpoint(filepath, cuda)

    return vae, vae_env_config['sound_normalization']


@ex.main
def main(_config, _run):
    os.makedirs(log_dir_path('trained_models'), exist_ok=True)
    os.makedirs(log_dir_path('results'), exist_ok=True)

    vae, sound_normalization = get_vae_model(
        _config, _config['gpu']['cuda'])
    vae.eval()
    train(vae, sound_normalization)


if __name__ == '__main__':
    ex.run_commandline()