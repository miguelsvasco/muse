import os
import sacred
import muse.scenarios.multimodal_atari.hyperhot_env as hh
import muse.evaluation.atari.hyperhot.rl.utils as dqn_utils
import muse.evaluation.atari.hyperhot.vae.utils as vae_utils
from muse.evaluation.atari.hyperhot.rl.eval import DQNEvaluator
import muse.evaluation.atari.hyperhot.ingredients as ingredients
from muse.evaluation.atari.hyperhot.hyperhot_processor import Processor
from muse.scenarios.multimodal_atari.hyperhot_env import HyperhotEnv

ex = sacred.Experiment(
    'hyperhot_muse_dqn_eval_pipeline',
    ingredients=[
        ingredients.hyperhot_ingredient, ingredients.dqn_ingredient,
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


def hyperhot_environment_config_matches(config, rhang_vae_env_config):
    # checks if the config dict is inside the rhang_vae_env_config dict
    return config.items() <= rhang_vae_env_config.items()


def get_vae_model(config, cuda):

    vae_dep_config = config['eval_pipeline']

    vae, _, vae_env_config = vae_utils.load_checkpoint(vae_dep_config['vae_file'], cuda)

    return vae, vae_env_config['sound_normalization']


def get_dqn_model(config, cuda):
    eval_pipeline_config = config['eval_pipeline']

    dqn, _ = dqn_utils.load_checkpoint(eval_pipeline_config['dqn_file'], cuda)

    return dqn


@ex.capture
def post_episode_cb(info, _run):
    episode_number = info['episode_number']
    obs_mod = info['obs_mod']

    _run.log_scalar('total_reward', info['total_reward'], step=episode_number)
    _run.log_scalar(
        'discounted_reward', info['discounted_reward'], step=episode_number)


@ex.capture
def eval(vae, sound_normalization, dqn, _config, _run):
    vae.eval()
    dqn.eval()
    eval_pipeline_config = _config['eval_pipeline']
    dqn_config = _config['dqn']
    hyperhot_config = _config['hyperhot']

    env = HyperhotEnv(
        num_enemies=hyperhot_config['n_enemies'],
        pacifist_mode=hyperhot_config['pacifist_mode'],
        time_limit=hyperhot_config['time_limit'],
        sound_receivers=[
            hh.SoundReceiver(hh.SoundReceiver.Location[ss])
            for ss in hyperhot_config['sound_receivers']
        ])

    condition_on_image = eval_pipeline_config['condition_on_image']
    condition_on_sound = eval_pipeline_config['condition_on_sound']
    condition_on_joint = eval_pipeline_config['condition_on_joint']
    processor = Processor(vae, sound_normalization, _config['gpu']['cuda'])

    if condition_on_image:
        print("Using only image observations")
        obs_mod = 'image'
        preprocess = processor.preprocess_just_image
        postprocess = processor.postprocess_just_image
    elif condition_on_sound:
        print("Using only sound observations")
        obs_mod = 'sound'
        preprocess = processor.preprocess_just_sound
        postprocess = processor.postprocess_just_sound
    elif condition_on_joint:
        print("Using joint modality observations")
        obs_mod = 'joint'
        preprocess = processor.preprocess_joint
        postprocess = processor.postprocess_joint
    else:
        print(f'Unknown option. Should not happen')
        import sys
        sys.exit()

    evaluator = DQNEvaluator(dqn, env, obs_mod, _config['hyperhot']['n_stack'],
                             _config['gpu']['cuda'], preprocess, postprocess)
    evaluator.eval(eval_pipeline_config['eval_episodes'],
                   _config['dqn']['gamma'], post_episode_cb)


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
    dqn = get_dqn_model(_config, cuda)

    eval(vae, sound_normalization, dqn)


if __name__ == '__main__':
    ex.run_commandline()