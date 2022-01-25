import os
import torch
import sacred
import random
import numpy as np
from tqdm import tqdm
from muse.utils.utils import AverageMeter
import muse.evaluation.standard.celeb.ingredients as ingredients
import muse.evaluation.standard.celeb.model.training_utils as t_utils
import muse.evaluation.standard.celeb.model.evaluation_utils as e_utils
from muse.scenarios.standard_dataset.standard_dataset import StandardDataset

ex = sacred.Experiment(
    'celeb_muse_evaluate_likelihoods',
    ingredients=[
        ingredients.training_ingredient, ingredients.model_ingredient,
        ingredients.model_debug_ingredient, ingredients.gpu_ingredient,
        ingredients.evaluation_ingredient
    ])


@ex.capture
def log_dir_path(folder, _config):
    return os.path.join('.', f'results/log_{_config["seed"]}/', folder)

@ex.capture
def exp_dir_path(folder, _config):
    return os.path.join('.', folder)

@ex.capture
def get_model_by_config(config, cuda):
    model_evaluation_config = config['evaluation']
    model, _ = t_utils.load_checkpoint(model_evaluation_config['file_local'], cuda)
    return model


@ex.capture
def evaluate_marginal_img(model, dataset, n_samples, cuda, _config, _run):

    log_marginal_img_meter = AverageMeter()

    with torch.no_grad():

        for _, data in enumerate(tqdm(dataset)):

            img_data = data[0]
            img_data = img_data.repeat_interleave(repeats=n_samples, dim=0)

            if cuda:
                img_data = img_data.cuda()

            out, z, mu, logvar = model.img_vae(img_data)
            marginal_img = e_utils.log_marginal_img_estimate(recon_image=out,
                                                                 image=img_data,
                                                                 z=z,
                                                                 mu=mu,
                                                                 logvar=logvar,
                                                                 batch_size=data[0].size(0),
                                                                 n_samples=n_samples)
            log_marginal_img_meter.update(marginal_img.item())

    # Save results
    print("Log p(image) = " + str(log_marginal_img_meter.avg))
    _run.log_scalar('Log P(image)', log_marginal_img_meter.avg)

    return log_marginal_img_meter.avg


@ex.capture
def evaluate_marginal_label(model, dataset, n_samples, cuda, _config, _run):
    log_marginal_label_meter = AverageMeter()

    with torch.no_grad():

        for _, data in enumerate(tqdm(dataset)):

            label_data = data[1].float()
            label_data = label_data.repeat_interleave(repeats=n_samples, dim=0)

            if cuda:
                label_data = label_data.cuda()

            out, z, mu, logvar = model.label_vae(label_data)

            marginal_label = e_utils.log_marginal_label_estimate(recon_label=out,
                                                                 label=label_data,
                                                                 z=z,
                                                                 mu=mu,
                                                                 logvar=logvar,
                                                                 batch_size=data[0].size(0),
                                                                 n_samples=n_samples)
            log_marginal_label_meter.update(marginal_label.item())

    # Save results
    print("Log p(attribute) = " + str(log_marginal_label_meter.avg))
    _run.log_scalar('Log P(attribute)', log_marginal_label_meter.avg)

    return log_marginal_label_meter.avg


@ex.capture
def evaluate_joint(model, dataset, n_samples, cuda, _config, _run):
    log_joint_meter = AverageMeter()

    with torch.no_grad():

        for _, data in enumerate(tqdm(dataset)):

            img_data = data[0]
            img_data = img_data.repeat_interleave(repeats=n_samples, dim=0)
            label_data = data[1].float()
            label_data = label_data.repeat_interleave(repeats=n_samples, dim=0)

            if cuda:
                img_data = img_data.cuda()
                label_data = label_data.cuda()

            z, mu, logvar = model.encode_latent(x_img=img_data, x_label=label_data)

            out_img = model.img_vae.decoder(model.img_top_decoder(z))
            out_label = model.label_vae.decoder(model.label_top_decoder(z))


            joint_estimate = e_utils.log_joint_estimate(recon_image=out_img,
                                                        image=img_data,
                                                        recon_label=out_label,
                                                        label=label_data,
                                                        z=z,
                                                        mu=mu,
                                                        logvar=logvar,
                                                        batch_size=data[0].size(0),
                                                        n_samples=n_samples)

            log_joint_meter.update(joint_estimate.item())

    # Save results
    print("Log p(image, attribute) = " + str(log_joint_meter.avg))
    _run.log_scalar('Log P(Joint)', log_joint_meter.avg)

    return log_joint_meter.avg


@ex.capture
def evaluate_conditional_img(model, dataset, n_samples, cuda, _config, _run):
    log_conditional_img_meter = AverageMeter()

    with torch.no_grad():

        for _, data in enumerate(tqdm(dataset)):

            img_data = data[0]
            img_data = img_data.repeat_interleave(repeats=n_samples, dim=0)
            label_data = data[1].float()
            label_data = label_data.repeat_interleave(repeats=n_samples, dim=0)

            if cuda:
                img_data = img_data.cuda()
                label_data = label_data.cuda()

            z, mu, logvar = model.encode_latent(x_img=None, x_label=label_data)

            z_img = model.img_top_decoder(z)
            out = model.img_vae.decoder(z_img)

            conditional_img_estimate = e_utils.log_conditional_image_estimate(recon_image=out,
                                                        image=img_data,
                                                        z=z,
                                                        mu=mu,
                                                        logvar=logvar,
                                                        batch_size=data[0].size(0),
                                                        n_samples=n_samples)

            log_conditional_img_meter.update(conditional_img_estimate.item())

    # Save results
    print("Log p(image | attribute) = " + str(log_conditional_img_meter.avg))
    _run.log_scalar('Log P(image | attribute)', log_conditional_img_meter.avg)

    return log_conditional_img_meter.avg



@ex.capture
def evaluate_conditional_label(model, dataset, n_samples, cuda, _config, _run):
    log_conditional_label_meter = AverageMeter()

    with torch.no_grad():

        for _, data in enumerate(tqdm(dataset)):

            img_data = data[0]
            img_data = img_data.repeat_interleave(repeats=n_samples, dim=0)
            label_data = data[1].float()
            label_data = label_data.repeat_interleave(repeats=n_samples, dim=0)

            if cuda:
                img_data = img_data.cuda()
                label_data = label_data.cuda()

            z, mu, logvar = model.encode_latent(x_img=img_data, x_label=None)
            z_top_label = model.label_top_decoder(z)
            out = model.label_vae.decoder(z_top_label)

            conditional_label_estimate = e_utils.log_conditional_label_estimate(recon_label=out,
                                                        label=label_data,
                                                        z=z,
                                                        mu=mu,
                                                        logvar=logvar,
                                                        batch_size=data[0].size(0),
                                                        n_samples=n_samples)

            log_conditional_label_meter.update(conditional_label_estimate.item())

    # Save results
    print("Log p(attribute | image) = " + str(log_conditional_label_meter.avg))
    _run.log_scalar('Log P(attribute | image)', log_conditional_label_meter.avg)

    return log_conditional_label_meter.avg




@ex.main
def main(_config, _run):

    os.makedirs(exp_dir_path('evaluation'), exist_ok=True)

    # Set seeds
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(_config['training']['seed'])
    np.random.seed(_config['training']['seed'])
    random.seed(_config['training']['seed'])
    torch.cuda.manual_seed(_config['training']['seed'])

    # Load model
    print("Loading model...")
    model = get_model_by_config(_config, _config['gpu']['cuda'])
    model.eval()

    # Load dataset
    dataset = StandardDataset(
        dataset='celeb',
        data_dir='../../../scenarios/standard_dataset/data',
        seed=_config['training']['seed'])
    eval_dataset = dataset.get_test_loader(bsize=1)

    # Evaluation Setup
    cuda = _config['gpu']['cuda']
    n_samples = _config['evaluation']['eval_samples']
    _run.log_scalar('eval_samples', n_samples)

    # Evaluate
    log_px1 = evaluate_marginal_img(model, eval_dataset, n_samples, cuda, _config)
    log_px2 = evaluate_marginal_label(model, eval_dataset, n_samples, cuda, _config)
    log_joint = evaluate_joint(model, eval_dataset, n_samples, cuda, _config)
    log_px1_gx2 = evaluate_conditional_img(model, eval_dataset, n_samples, cuda, _config)
    log_px2_gx1 = evaluate_conditional_label(model, eval_dataset, n_samples, cuda, _config)

    # Save
    with open(os.path.join(exp_dir_path('evaluation'), "likelihoods.txt"), 'w') as f:
        print('\nResults:\nLog P(x1): ' + str(log_px1) + '\n'
              + 'Log P(x2): ' + str(log_px2) + '\n'
              + 'Log Joint: ' + str(log_joint) + '\n'
              + 'Log P(x1|x2): ' + str(log_px1_gx2) + '\n'
              + 'Log P(x2|x1): ' + str(log_px2_gx1) + '\n', file=f)

    ex.add_artifact(os.path.join(exp_dir_path('evaluation'),
                                 "likelihoods.txt"), name='likelihoods.txt')



if __name__ == '__main__':
    ex.run_commandline()
