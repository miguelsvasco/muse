import os
import torch
import shutil
from muse.evaluation.atari.hyperhot.rl.model import DQN


def save_checkpoint(state,
                    is_best,
                    folder='./',
                    filename='dqn_checkpoint.pth.tar',
                    best_filename='best_dqn_model.pth.tar'):
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(folder, filename), os.path.join(
                folder, best_filename))


def load_checkpoint(checkpoint_file, use_cuda=False):
    if use_cuda:
        checkpoint = torch.load(checkpoint_file)
    else:
        checkpoint = torch.load(checkpoint_file, map_location='cpu')

    dqn_config = checkpoint['dqn_config']
    env_config = checkpoint['env_config']
    dqn = DQN(env_config['n_states'], env_config['n_actions'],
              dqn_config['layers_sizes'], use_cuda)
    dqn.load_state_dict(checkpoint['state_dict'])

    return dqn, dqn_config