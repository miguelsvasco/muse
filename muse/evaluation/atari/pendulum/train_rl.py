import os
import sacred
import imageio
import numpy as np
from gym.wrappers.time_limit import TimeLimit as TimeLimitWrapper
import muse.scenarios.multimodal_atari.pendulum_sound as ps
from muse.evaluation.atari.pendulum.rl.model import DDPG
import muse.evaluation.atari.pendulum.vae.utils as vae_utils
import muse.evaluation.atari.pendulum.rl.utils as ddpg_utils
import muse.evaluation.atari.pendulum.ingredients as ingredients
from muse.evaluation.atari.pendulum.rl.trainer import DDPGTrainer
from muse.scenarios.multimodal_atari.pendulum_sound import PendulumSound
from muse.evaluation.atari.pendulum.pendulum_processor import Processor


ex = sacred.Experiment(
    'pendulum_muse_ddpg_train_ddpg',
    ingredients=[
        ingredients.pendulum_ingredient, ingredients.ddpg_ingredient,
        ingredients.ddpg_eval_ingredient, ingredients.vae_ingredient,
        ingredients.ddpg_setup_vae_dep_ingredient, ingredients.gpu_ingredient
    ])


@ex.capture
def log_dir_path(folder, _config):
    return os.path.join('./results', f'log_{_config["seed"]}', folder)


@ex.capture
def post_episode_cb(info, _run):
    frame_number = info['frame_number']

    _run.log_scalar(
        'avg_critic_loss', info['avg_critic_loss'], step=frame_number)
    _run.log_scalar(
        'avg_actor_loss', info['avg_actor_loss'], step=frame_number)
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
    def __init__(self, ddpg, actor_optimizer, critic_optimizer):
        self.ddpg = ddpg
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
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
        frame_shape = info['eval_observations'][0][0][0].shape
        black_frames = [
            np.zeros(frame_shape, dtype=np.uint8) for _ in range(frame_rate)
        ]
        white_frames = [
            np.ones(frame_shape, dtype=np.uint8) * 255
            for _ in range(2 * frame_rate)
        ]

        gif_path = os.path.join(
            log_dir_path('results'), f'pendulum_eval_{frame_number}.gif')
        frames_for_gif = []
        for episode_observations in info['eval_observations']:
            frames_for_gif.extend(
                [image for (image, sound) in episode_observations])
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
            'n_states': self.ddpg.n_states,
            'n_actions': self.ddpg.n_actions
        }
        # TODO: Find a better way to do this. _config is a
        # ReadOnlyDict defined by sacred, which is not pickleable. And
        # due to the nested config, we can't simply do dict(_config)
        pickleable_ddpg_config = dict(_config['ddpg'])
        pickleable_ddpg_config['random_process'] = dict(
            pickleable_ddpg_config['random_process'])
        ddpg_utils.save_checkpoint(
            {
                'state_dict': self.ddpg.state_dict(),
                'best_reward': self.best_reward,
                'frame_number': info['frame_number'],
                'actor_optimizer': self.actor_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict(),
                'ddpg_config': pickleable_ddpg_config,
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
def train(rhang_vae, sound_normalization, _config, _run):
    pendulum_config = _config['pendulum']
    ddpg_config = _config['ddpg']
    ddpg_eval_config = _config['ddpg_eval']

    env = PendulumSound(
        pendulum_config['original_frequency'],
        pendulum_config['sound_velocity'],
        sound_receivers=[
            ps.SoundReceiver(ps.SoundReceiver.Location[ss])
            for ss in pendulum_config['sound_receivers']
        ])
    if ddpg_config['max_episode_length'] > 0:
        env = TimeLimitWrapper(env, ddpg_config['max_episode_length'])

    ddpg = DDPG(rhang_vae.n_latents, env.action_space.shape[0],
                ddpg_config['actor_layers_sizes'][0],
                ddpg_config['actor_layers_sizes'][1], _config['gpu']['cuda'])

    processor = Processor(rhang_vae, sound_normalization,
                          _config['gpu']['cuda'])
    if ddpg_config['condition_on_image']:
        print('DDPG conditioned on image.')
        preprocess = processor.preprocess_just_image
        postprocess = processor.postprocess_just_image
    elif ddpg_config['condition_on_sound']:
        print('DDPG conditioned on sound.')
        preprocess = processor.preprocess_just_sound
        postprocess = processor.postprocess_just_sound
    elif ddpg_config['condition_on_joint']:
        print('DDPG conditioned on joint.')
        preprocess = processor.preprocess_joint
        postprocess = processor.postprocess_joint

    else:
        raise 'Unknown conditioning option'

    trainer = DDPGTrainer(
        ddpg, env, pendulum_config['n_stack'], ddpg_config['gamma'],
        ddpg_config['actor_learning_rate'],
        ddpg_config['critic_learning_rate'], ddpg_config['tau'],
        ddpg_config['batch_size'], ddpg_config['memory_size'],
        ddpg_config['random_process'], _config['gpu']['cuda'], preprocess,
        postprocess)

    post_eval_cb = PostEvalCb(ddpg, trainer.actor_optim, trainer.critic_optim)
    trainer.train(
        ddpg_config['max_frames'], ddpg_eval_config['eval_frequency'],
        ddpg_eval_config['eval_length'], post_episode_cb, post_eval_cb)

    ex.add_artifact(
        os.path.join(
            log_dir_path('trained_models'), 'ddpg_checkpoint.pth.tar'),
        name='ddpg_last_checkpoint.pth.tar')
    ex.add_artifact(
        os.path.join(
            log_dir_path('trained_models'), 'best_ddpg_model.pth.tar'))

    env.close()


def pendulum_environment_config_matches(config, rhang_vae_env_config):
    # TODO(rsilva)
    print(f'This is yet to be implemented. Saying yes for now...')
    return True


def flatten_dict(dd, separator='_', prefix=''):
    return {
        prefix + separator + k if prefix else k: v
        for kk, vv in dd.items()
        for k, v in flatten_dict(vv, separator, kk).items()
    } if isinstance(dd, dict) else {
        prefix: dd
    }


def get_vae_model(config, cuda):
    vae_dep_config = config['ddpg_setup_vae_dep']

    vae, _, vae_env_config = vae_utils.load_checkpoint(
        vae_dep_config['file'], cuda)

    return vae, vae_env_config['sound_normalization']


@ex.main
def main(_config, _run):
    os.makedirs(log_dir_path('trained_models'), exist_ok=True)
    os.makedirs(log_dir_path('results'), exist_ok=True)
    vae, sound_normalization = get_vae_model(_config, _config['gpu']['cuda'])
    vae.eval()
    train(vae, sound_normalization)


if __name__ == '__main__':
    ex.run_commandline()