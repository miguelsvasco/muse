import os
import torch
import shutil
from muse.evaluation.standard.mnist.model.model import MUSE

def save_checkpoint(state,
                    is_best,
                    folder='./',
                    filename='muse_checkpoint.pth.tar'):
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(folder, filename),
            os.path.join(folder, 'best_muse_model.pth.tar'))


def load_checkpoint(checkpoint_file, use_cuda=False):
    if use_cuda:
        checkpoint = torch.load(checkpoint_file)
    else:
        checkpoint = torch.load(
            checkpoint_file, map_location=lambda storage, location: storage)

    model_config = checkpoint['model_config']
    model = MUSE(img_latents=model_config['mnist_latent_dim'],
                  label_latents=model_config['label_latent_dim'],
                  top_latents=model_config['top_latent_dim'],
                  use_cuda=use_cuda)
    model.load_state_dict(checkpoint['state_dict'])

    if use_cuda:
        model.cuda()

    return model, checkpoint['training_config']

