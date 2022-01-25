import os
import shutil
import torch
from muse.evaluation.atari.pendulum.rl.model import DDPG


def save_checkpoint(state,
                    is_best,
                    folder='./',
                    filename='ddpg_checkpoint.pth.tar'):
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(folder, filename),
            os.path.join(folder, 'best_ddpg_model.pth.tar'))


def load_checkpoint(checkpoint_file, use_cuda=False):
    if use_cuda:
        checkpoint = torch.load(checkpoint_file)
    else:
        checkpoint = torch.load(checkpoint_file, map_location='cpu')

    ddpg_config = checkpoint['ddpg_config']
    env_config = checkpoint['env_config']
    ddpg = DDPG(env_config['n_states'], env_config['n_actions'],
                ddpg_config['actor_layers_sizes'][0],
                ddpg_config['actor_layers_sizes'][1], use_cuda)
    ddpg.load_state_dict(checkpoint['state_dict'])

    return ddpg, ddpg_config