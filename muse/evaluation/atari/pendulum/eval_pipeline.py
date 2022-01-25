import os
import sacred
import numpy as np
import muse.scenarios.multimodal_atari.pendulum_sound as ps
import muse.evaluation.atari.pendulum.vae.utils as vae_utils
import muse.evaluation.atari.pendulum.rl.utils as ddpg_utils
from muse.evaluation.atari.pendulum.rl.eval import DDPGEvaluator
import muse.evaluation.atari.pendulum.ingredients as ingredients
from gym.wrappers.time_limit import TimeLimit as TimeLimitWrapper
from muse.evaluation.atari.pendulum.pendulum_processor import  Processor
from muse.scenarios.multimodal_atari.pendulum_sound import PendulumSound


ex = sacred.Experiment(
    'pendulum_muse_ddpg_eval_pipeline',
    ingredients=[
        ingredients.pendulum_ingredient, ingredients.ddpg_ingredient,
        ingredients.vae_ingredient, ingredients.eval_pipeline_ingredient,
        ingredients.gpu_ingredient
    ])


@ex.capture
def log_dir_path(folder, _config):
    return os.path.join('./results', f'log_{_config["seed"]}', folder)


def flatten_dict(dd, separator='_', prefix=''):
    return {
        prefix + separator + k if prefix else k: v
        for kk, vv in dd.items()
        for k, v in flatten_dict(vv, separator, kk).items()
    } if isinstance(dd, dict) else {
        prefix: dd
    }


def get_vae_model(config, cuda):

    vae_dep_config = config['eval_pipeline']

    vae, _, vae_env_config = vae_utils.load_checkpoint(vae_dep_config['vae_file'], cuda)

    return vae, vae_env_config['sound_normalization']


def get_ddpg_model(config, cuda):
    eval_pipeline_config = config['eval_pipeline']

    ddpg, _ = ddpg_utils.load_checkpoint(eval_pipeline_config['ddpg_file'], cuda)

    return ddpg


@ex.capture
def post_episode_cb(info, _run):
    episode_number = info['episode_number']
    obs_mod = info['obs_mod']
    _run.log_scalar('avg_reward', info['avg_reward'], step=episode_number)

def sine(amplitude=1.0, frequency=440.0, duration=1.0):
    points = int(31400 * duration)
    times = np.linspace(0, duration, points, endpoint=False)
    data = np.array(np.sin(times * frequency * 2 * np.pi) * amplitude)
    sound = np.array(data * 32767).astype(np.int16)
    return sound



@ex.capture
def eval(vae, sound_normalization, ddpg, _config, _run):
    vae.eval()
    ddpg.eval()

    eval_pipeline_config = _config['eval_pipeline']
    ddpg_config = _config['ddpg']
    pendulum_config = _config['pendulum']

    env = PendulumSound(
        pendulum_config['original_frequency'],
        pendulum_config['sound_velocity'],
        sound_receivers=[
            ps.SoundReceiver(ps.SoundReceiver.Location[ss])
            for ss in pendulum_config['sound_receivers']
        ])
    if ddpg_config['max_episode_length'] > 0:
        env = TimeLimitWrapper(env, ddpg_config['max_episode_length'])

    condition_on_image = eval_pipeline_config['condition_on_image']
    condition_on_sound = eval_pipeline_config['condition_on_sound']
    condition_on_joint = eval_pipeline_config['condition_on_joint']
    processor = Processor(vae, sound_normalization, _config['gpu']['cuda'])

    if condition_on_image:
        print("Using only image observations")
        obs_mod='image'
        preprocess = processor.preprocess_just_image
        postprocess = processor.postprocess_just_image
    elif condition_on_sound:
        print("Using only sound observations")
        obs_mod='sound'
        preprocess = processor.preprocess_just_sound
        postprocess = processor.postprocess_just_sound
    elif condition_on_joint:
        print("Using joint modality observations")
        obs_mod='joint'
        preprocess = processor.preprocess_joint
        postprocess = processor.postprocess_joint
    else:
        print(f'Unknown option. Should not happen')
        import sys
        sys.exit()

    evaluator = DDPGEvaluator(ddpg, env, obs_mod,_config['pendulum']['n_stack'],
                              _config['gpu']['cuda'], preprocess, postprocess)

    evaluator.eval(eval_pipeline_config['eval_episodes'],
                   eval_pipeline_config['eval_episode_length'],
                   post_episode_cb)


@ex.main
def main(_config):

    # Create folders

    eval_pipeline_config = _config['eval_pipeline']
    condition_on_image = eval_pipeline_config['condition_on_image']
    condition_on_joint = eval_pipeline_config['condition_on_joint']

    if condition_on_joint:
        os.makedirs(log_dir_path('results_joint'), exist_ok=True)
    elif condition_on_image:
        os.makedirs(log_dir_path('results_image'), exist_ok=True)
    else:
        os.makedirs(log_dir_path('results_sound'), exist_ok=True)

    cuda = _config['gpu']['cuda']
    vae, sound_normalization = get_vae_model(_config, cuda)
    ddpg = get_ddpg_model(_config, cuda)

    eval(vae, sound_normalization, ddpg)


if __name__ == '__main__':
    ex.run_commandline()
