import os
import torch
import sacred
from torchvision.utils import save_image
import muse.evaluation.standard.celeb.ingredients as ingredients
import muse.evaluation.standard.celeb.model.training_utils as utils

"""
CelebA attributes

0-5_o_Clock_Shadow
1-Arched_Eyebrows
2-Attractive
3-Bags_Under_Eyes
4-Bald
5-Bangs
6-Big_Lips
7-Big_Nose
8-Black_Hair
9-Blond_Hair
10-Blurry
11-Brown_Hair
12-Bushy_Eyebrows
13-Chubby
14-Double_Chin
15-Eyeglasses
16-Goatee
17-Gray_Hair
18-Heavy_Makeup
19-High_Cheekbones
20-Male
21-Mouth_Slightly_Open
22-Mustache
23-Narrow_Eyes
24-No_Beard
25-Oval_Face
26-Pale_Skin
27-Pointy_Nose
28-Receding_Hairline
29-Rosy_Cheeks
30-Sideburns
31-Smiling
32-Straight_Hair
33-Wavy_Hair
34-Wearing_Earrings
35-Wearing_Hat
36-Wearing_Lipstick
37-Wearing_Necklace
38-Wearing_Necktie
39-Young
"""


ex = sacred.Experiment(
    'celeb_muse_generation',
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
    sample = generate_samples(model, n_samples, cuda)
    save_image(sample.view(-1, 3, 64, 64).cpu(),
               os.path.join(results_dir, 'image_samples.png'))
    ex.add_artifact(
        os.path.join(results_dir, 'image_samples.png'),
        name='image_samples.png')



def generate_samples(model, n_samples, cuda):

    attributes = torch.zeros(40).float()

    # MR SMITH evaluation
    attributes[8]=1.0  # Black hair
    attributes[15]=1.0 # Eyeglasses
    attributes[16]=1.0 # Goatee
    attributes[20]=1.0 # Male
    attributes[28]=1.0 # Receading Hairline
    attributes[32]=1.0 # Straight_Hair

    # MS SMITH evaluation
    # attributes[2]=1.0
    # attributes[6]=1.0
    # attributes[9]=1.0
    # attributes[18]=1.0
    # attributes[24]=1.0
    # attributes[31]=1.0
    # attributes[33]=1.0
    # attributes[36]=1.0
    # attributes[39]=1.0

    # Granny SMITH evaluation
    # attributes[3]=1.0
    # attributes[7]=1.0
    # attributes[12]=1.0
    # attributes[17]=1.0
    # attributes[21]=1.0
    # attributes[23]=1.0
    # attributes[24]=1.0
    # attributes[35]=1.0
    # attributes[33]=1.0
    # attributes[34]=1.0

    attributes = attributes.unsqueeze(0)
    attributes = attributes.repeat(n_samples, 1)

    if cuda:
        attributes = attributes.cuda()

    # Encode Mod Latents
    m_out, _= model.generate(x_label=attributes)

    return m_out[1]



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
