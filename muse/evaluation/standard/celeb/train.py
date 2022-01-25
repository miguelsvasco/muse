import os
import sys
import sacred
import random
import torch
import torchvision
import numpy as np
from muse.evaluation.standard.celeb.model.model import MUSE
import muse.evaluation.standard.celeb.ingredients as ingredients
from muse.evaluation.standard.celeb.model.trainer import Trainer
import muse.evaluation.standard.celeb.model.training_utils as t_utils
from muse.scenarios.standard_dataset.standard_dataset import StandardDataset

ex = sacred.Experiment(
    'celeb_muse_train',
    ingredients=[
        ingredients.training_ingredient, ingredients.model_ingredient,
        ingredients.model_debug_ingredient, ingredients.gpu_ingredient
    ])

@ex.capture
def log_dir_path(folder, _config):
    return os.path.join('.', f'results/log_{_config["seed"]}', folder)


@ex.capture
def exp_dir_path(folder, _config):
    return os.path.join('.', folder)


class PostEpochCb(object):
    def __init__(self, model, dataset):
        self.model = model
        self.train_dataloader = dataset.train_loader
        self.val_dataloader = dataset.val_loader
        self.test_dataloader = dataset.get_test_loader(bsize=20)
        self.best_loss = sys.maxsize

    @ex.capture
    def _record_train_info(self, info, _run):
        _run.log_scalar('train_loss', info['loss'])
        _run.log_scalar('train_img_recon_loss', info['img_recon_loss'])
        _run.log_scalar('train_img_prior_loss', info['img_prior_loss'])
        _run.log_scalar('train_label_recon_loss', info['label_recon_loss'])
        _run.log_scalar('train_label_prior_loss', info['label_prior_loss'])
        _run.log_scalar('train_img_top_recon_loss', info['img_top_recon_loss'])
        _run.log_scalar('train_label_top_recon_loss', info['label_top_recon_loss'])
        _run.log_scalar('train_top_prior_loss', info['top_prior_loss'])
        _run.log_scalar('train_top_fpa_loss', info['top_fpa_loss'])

    @ex.capture
    def _record_test_info(self, info, _run):
        _run.log_scalar('test_loss', info['loss'])
        _run.log_scalar('test_img_recon_loss', info['img_recon_loss'])
        _run.log_scalar('test_img_prior_loss', info['img_prior_loss'])
        _run.log_scalar('test_label_recon_loss', info['label_recon_loss'])
        _run.log_scalar('test_label_prior_loss', info['label_prior_loss'])
        _run.log_scalar('test_img_top_recon_loss', info['img_top_recon_loss'])
        _run.log_scalar('test_label_top_recon_loss', info['label_top_recon_loss'])
        _run.log_scalar('test_top_prior_loss', info['top_prior_loss'])
        _run.log_scalar('test_top_fpa_loss', info['top_fpa_loss'])

    @ex.capture
    def _record_artifacts(self, info, _config):
        epoch = info['epoch']
        artifact_storage_interval = _config['model_debug'][
            'artifact_storage_interval']
        results_dir = log_dir_path('results')

        if epoch % artifact_storage_interval == 0:

            # Data
            with torch.no_grad():

                self.model.eval()
                test_data = next(iter(self.test_dataloader))
                img_data = test_data[0]
                label_data = test_data[1].float()

                if self.model.use_cuda:
                    img_data = img_data.cuda()
                    label_data = label_data.cuda()

                # Generate modalities
                img_out, label_out  = self.model.generate(x_img=img_data,x_label=label_data)
                img_out, img_top_out = img_out[0], img_out[1]

                img_top_img_out, img_top_label_out = self.model.generate(x_img=img_data)
                img_top_img_out = img_top_img_out[1]

                label_top_img_out, label_top_label_out = self.model.generate(x_label=label_data)
                label_top_img_out = label_top_img_out[1]

                # Vision Recon
                img_comp = torch.cat([img_data.view(-1, 3, 64, 64).cpu(), img_out.view(-1, 3, 64, 64).cpu()])
                all_top_img_comp = torch.cat([img_data.view(-1, 3, 64, 64).cpu(), img_top_out.view(-1, 3, 64, 64).cpu()])
                img_top_img_comp = torch.cat([img_data.view(-1, 3, 64, 64).cpu(), img_top_img_out.view(-1, 3, 64, 64).cpu()])
                sym_top_img_comp = torch.cat([img_data.view(-1, 3, 64, 64).cpu(), label_top_img_out.view(-1, 3, 64, 64).cpu()])


            # Save data
            torchvision.utils.save_image(torchvision.utils.make_grid(img_comp,
                                                                     padding=5,
                                                                     pad_value=.5,
                                                                     nrow=img_data.size(0)),
                                         os.path.join(results_dir, 'img_mod_e' + str(epoch) + '.png'))
            ex.add_artifact(os.path.join(results_dir, "img_mod_e" + str(epoch) + '.png'),
                            name="image_recon_e" + str(epoch) + '.png')

            torchvision.utils.save_image(torchvision.utils.make_grid(all_top_img_comp,
                                                                     padding=5,
                                                                     pad_value=.5,
                                                                     nrow=img_data.size(0)),
                                         os.path.join(results_dir, 'all_top_img_comp_e' + str(epoch) + '.png'))
            ex.add_artifact(os.path.join(results_dir, "all_top_img_comp_e" + str(epoch) + '.png'),
                            name="all_top_image_comp_e" + str(epoch) + '.png')

            torchvision.utils.save_image(torchvision.utils.make_grid(img_top_img_comp,
                                                                     padding=5,
                                                                     pad_value=.5,
                                                                     nrow=img_data.size(0)),
                                         os.path.join(results_dir, 'img_top_img_comp_e' + str(epoch) + '.png'))
            ex.add_artifact(os.path.join(results_dir, "img_top_img_comp_e" + str(epoch) + '.png'),
                            name="image_top_image_comp_e" + str(epoch) + '.png')

            torchvision.utils.save_image(torchvision.utils.make_grid(sym_top_img_comp,
                                                                     padding=5,
                                                                     pad_value=.5,
                                                                     nrow=img_data.size(0)),
                                         os.path.join(results_dir, 'label_top_img_comp_e' + str(epoch) + '.png'))
            ex.add_artifact(os.path.join(results_dir, "label_top_img_comp_e" + str(epoch) + '.png'),
                            name="label_top_image_comp_e" + str(epoch) + '.png')



    @ex.capture
    def _record_checkpoint(self, info, _config):
        test_info = info['test']
        loss = test_info['loss']
        is_best = loss < self.best_loss
        self.best_loss = min(loss, self.best_loss)

        # Find a way to make this modality agnostic..... -- TODO
        model_config = dict(_config['model'])
        training_config = dict(_config['training'])

        t_utils.save_checkpoint(
            {
                'state_dict': self.model.state_dict(),
                'best_loss': self.best_loss,
                'loss': test_info['loss'],
                'epoch': info['epoch'],
                'optimizer': info['optimizer'].state_dict(),
                'model_config': model_config,
                'training_config': training_config
            },
            is_best,
            folder=log_dir_path('trained_models'))

    @ex.capture
    def __call__(self, epoch_info, _config):
        self._record_train_info(epoch_info['train'])
        self._record_test_info(epoch_info['test'])
        self._record_artifacts(epoch_info)
        self._record_checkpoint(epoch_info)


def post_cb(info):
    ex.add_artifact(
        os.path.join(
            log_dir_path('trained_models'), 'muse_checkpoint.pth.tar'),
        name='muse_last_checkpoint.pth.tar')
    ex.add_artifact(
        os.path.join(
            log_dir_path('trained_models'), 'best_muse_model.pth.tar'),
        name='best_muse_model.pth.tar')


@ex.capture
def train(_config, _run):

    # Read configs
    model_config = _config['model']
    training_config = _config['training']
    gpu_config = _config['gpu']

    # Set seeds
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(training_config['seed'])
    np.random.seed(training_config['seed'])
    random.seed(training_config['seed'])
    torch.cuda.manual_seed(training_config['seed'])

    # Create Model
    model = MUSE(img_latents=model_config['img_latent_dim'],
                  label_latents=model_config['label_latent_dim'],
                  top_latents=model_config['top_latent_dim'],
                  use_cuda=gpu_config['cuda'])

    # Create trainer
    trainer = Trainer(model, training_config, gpu_config['cuda'])

    # Create Dataset
    dataset = StandardDataset(
        dataset='celeb',
        data_dir='../../../scenarios/standard_dataset/data',
        batch_size=training_config['batch_size'],
        eval_samples=training_config['eval_samples'],
        validation_size=training_config['validation_size'],
        seed=training_config['seed'])

    post_epoch_cb = PostEpochCb(model, dataset)

    trainer.train(epochs=training_config['epochs'], dataset=dataset, cuda=gpu_config['cuda'],
                  post_epoch_cb=post_epoch_cb, post_cb=post_cb)


@ex.main
def main(_config, _run):
    os.makedirs(log_dir_path('trained_models'), exist_ok=True)
    os.makedirs(log_dir_path('results'), exist_ok=True)

    train()


if __name__ == '__main__':
    ex.run_commandline()