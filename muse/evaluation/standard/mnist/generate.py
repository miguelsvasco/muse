import os
import torch
import sacred
from tqdm import tqdm
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import muse.evaluation.standard.mnist.ingredients as ingredients
import muse.evaluation.standard.mnist.model.training_utils as utils

ex = sacred.Experiment(
    'mnist_muse_generation',
    ingredients=[
        ingredients.training_ingredient, ingredients.model_ingredient,
        ingredients.model_debug_ingredient, ingredients.gpu_ingredient,
        ingredients.evaluation_ingredient,
        ingredients.generation_ingredient
    ])


@ex.capture
def log_dir_path(folder, _config):
    return os.path.join('.', f'results/log_{_config["seed"]}/', folder)

@ex.capture
def exp_dir_path(folder, _config):
    return os.path.join('.', folder)



@ex.capture
def sample(model, _config, _run):
    model_eval_config = _config['generation']
    cuda = _config['gpu']['cuda']

    # Setup Generation
    results_dir = exp_dir_path('samples')
    n_samples = model_eval_config['n_samples']

    # Sample image from prior mod
    prior_mod = generate_prior_mod(model, n_samples, cuda)
    save_image(prior_mod.view(n_samples, 1, 28, 28),
               os.path.join(results_dir, 'image_prior_mod.png'))
    ex.add_artifact(
        os.path.join(results_dir, 'image_prior_mod.png'),
        name='image_prior_mod.png')

    # Sample image from prior top
    prior_mod = generate_prior_top_(model, n_samples, cuda)
    save_image(prior_mod.view(n_samples, 1, 28, 28),
               os.path.join(results_dir, 'image_prior_top_.png'))
    ex.add_artifact(
        os.path.join(results_dir, 'image_prior_top_.png'),
        name='image_prior_top_.png')

    # Sample cross-modality from symbol
    for i in tqdm(range(0, 10)):
        sample_v = generate_class(model, i, n_samples, cuda)
        # save image samples to filesystem
        save_image(sample_v.view(n_samples, 1, 28, 28),
                   os.path.join(results_dir, 'image_label_' + str(i) + '.png'))
        ex.add_artifact(
            os.path.join(results_dir, 'image_label_' + str(i) + '.png'),
            name='image_label_' + str(i) + '.png')



def generate_class(model, label_class, n_samples, cuda):

    symbol = F.one_hot(torch.tensor(label_class), 10).float().unsqueeze(0)
    symbol = symbol.repeat(n_samples, 1)

    if cuda:
        symbol = symbol.cuda()

    # Encode Mod Latents
    m_out, _= model.generate(x_label=symbol)

    return m_out[1]



def generate_prior_top_(model, n_samples, cuda):
    z = Variable(torch.randn(n_samples, model.top_latents))
    if cuda:
        z = z.cuda()

    # Decode
    mnist_top_z = model.mnist_top_decoder(z)
    mnist_out = model.mnist_vae.decoder(mnist_top_z)
    return mnist_out


def generate_prior_mod(model, n_samples, cuda):
    z = Variable(torch.randn(n_samples, model.img_latents))
    if cuda:
        z = z.cuda()

    # Decode
    m_out = model.mnist_vae.decoder(z)
    return m_out



def get_model_by_config(config, cuda):

    model_evaluation_config = config['evaluation']
    model, _ = utils.load_checkpoint(model_evaluation_config['file_local'], cuda)
    return model

@ex.main
def main(_config, _run):
    os.makedirs(exp_dir_path('samples'), exist_ok=True)
    model = get_model_by_config(_config, _config['gpu']['cuda'])
    model.eval()
    sample(model, _config)


if __name__ == '__main__':
    ex.run_commandline()