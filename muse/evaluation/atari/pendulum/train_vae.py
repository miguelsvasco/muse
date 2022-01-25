import os
import sys
import torch
import sacred
import random
import torchvision
import numpy as np
import muse.evaluation.atari.pendulum.vae.utils as vae_utils
import muse.evaluation.atari.pendulum.ingredients as ingredients
from muse.evaluation.atari.pendulum.vae.trainer import VAETrainer
from muse.scenarios.multimodal_atari.pendulum_sound_dataset import PendulumSoundDataset

ex = sacred.Experiment(
    'pendulum_muse_train_vae',
    ingredients=[
        ingredients.pendulum_ingredient, ingredients.vae_ingredient,
        ingredients.vae_debug_ingredient, ingredients.gpu_ingredient
    ])


@ex.capture
def log_dir_path(folder, _config):
    return os.path.join('./results', f'log_{_config["seed"]}', folder)


def make_dataloaders(train_dataset_samples, test_dataset_samples, n_stack,
                     original_frequency, sound_velocity, sound_receivers,
                     batch_size):
    train_loader = torch.utils.data.DataLoader(
        PendulumSoundDataset(
            '../../../scenarios/multimodal_atari/data',
            generate=False,
            n_samples=train_dataset_samples,
            n_stack=n_stack,
            original_frequency=original_frequency,
            sound_velocity=sound_velocity,
            sound_receivers=sound_receivers),
        batch_size=batch_size,
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        PendulumSoundDataset(
            '../../../scenarios/multimodal_atari/data',
            generate=False,
            n_samples=test_dataset_samples,
            n_stack=n_stack,
            original_frequency=original_frequency,
            sound_velocity=sound_velocity,
            sound_receivers=sound_receivers),
        batch_size=batch_size,
        shuffle=True)

    return train_loader, test_loader


class PostEpochCb(object):
    def __init__(self, vae, train_dataloader, test_dataloader):
        self.vae = vae
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.best_loss = sys.maxsize

    @ex.capture
    def _record_train_info(self, info, _run):
        _run.log_scalar('train_loss', info['loss'])
        _run.log_scalar('train_image_recon_loss', info['image_recon_loss'])
        _run.log_scalar('train_image_prior_loss', info['image_prior_loss'])
        _run.log_scalar('train_sound_recon_loss', info['sound_recon_loss'])
        _run.log_scalar('train_sound_prior_loss', info['sound_prior_loss'])
        _run.log_scalar('train_image_top_recon_loss', info['image_top_recon_loss'])
        _run.log_scalar('train_sound_top_recon_loss', info['sound_top_recon_loss'])
        _run.log_scalar('train_prior_loss', info['prior_loss'])
        _run.log_scalar('train_fpa_loss', info['fpa_loss'])

    @ex.capture
    def _record_test_info(self, info, _run):
        _run.log_scalar('test_loss', info['loss'])
        _run.log_scalar('test_image_recon_loss', info['image_recon_loss'])
        _run.log_scalar('test_image_prior_loss', info['image_prior_loss'])
        _run.log_scalar('test_sound_recon_loss', info['sound_recon_loss'])
        _run.log_scalar('test_sound_prior_loss', info['sound_prior_loss'])
        _run.log_scalar('test_image_top_recon_loss', info['image_top_recon_loss'])
        _run.log_scalar('test_sound_top_recon_loss', info['sound_top_recon_loss'])
        _run.log_scalar('test_prior_loss', info['prior_loss'])
        _run.log_scalar('test_fpa_loss', info['fpa_loss'])


    @ex.capture
    def _record_artifacts_maybe(self, info, _config):
        epoch = info['epoch']
        artifact_storage_interval = _config['vae_debug']['artifact_storage_interval']
        results_dir = log_dir_path('results')
        if epoch % artifact_storage_interval == 0:
            n_samples = 20

            self.vae.eval()
            loader = torch.utils.data.DataLoader(self.test_dataloader.dataset, batch_size=n_samples, shuffle=True)

            for (image, sound) in loader:
                break  # silly way of getting first element...

            if _config['gpu']['cuda']:
                image, sound = image.cuda(), sound.cuda()

            with torch.no_grad():

                img_recon, _ = self.vae.generate(image=image, sound=sound)
                img_img_recon, _ = self.vae.generate(image=image)
                snd_img_recon, _ = self.vae.generate(sound=sound)
                img_recon, img_img_recon, snd_img_recon = img_recon[1], img_img_recon[1], snd_img_recon[1]

                # Vision Recon
                img_comp = torch.cat([image.view(-1, 2*60, 60).cpu(), img_recon.view(-1, 2*60, 60).cpu()], dim=0).unsqueeze(1)
                img_img_comp = torch.cat([image.view(-1, 2*60, 60).cpu(), img_img_recon.view(-1, 2*60, 60).cpu()], dim=0).unsqueeze(1)
                snd_img_comp = torch.cat([image.view(-1, 2*60, 60).cpu(), snd_img_recon.view(-1, 2*60, 60).cpu()], dim=0).unsqueeze(1)

            # Save data
            torchvision.utils.save_image(torchvision.utils.make_grid(img_comp,
                                                                     padding=20,
                                                                     pad_value=.5,
                                                                     nrow=image.size(0)),
                                         os.path.join(results_dir, 'img_mod_e' + str(epoch) + '.png'))
            ex.add_artifact(os.path.join(results_dir, "img_mod_e" + str(epoch) + '.png'),
                            name="image_recon_e" + str(epoch) + '.png')

            torchvision.utils.save_image(torchvision.utils.make_grid(img_img_comp,
                                                                     padding=20,
                                                                     pad_value=.5,
                                                                     nrow=image.size(0)),
                                         os.path.join(results_dir, 'img_top_img_comp_e' + str(epoch) + '.png'))
            ex.add_artifact(os.path.join(results_dir, "img_top_img_comp_e" + str(epoch) + '.png'),
                            name="image_top_image_comp_e" + str(epoch) + '.png')

            torchvision.utils.save_image(torchvision.utils.make_grid(snd_img_comp,
                                                                     padding=20,
                                                                     pad_value=.5,
                                                                     nrow=image.size(0)),
                                         os.path.join(results_dir, 'sound_top_img_comp_e' + str(epoch) + '.png'))
            ex.add_artifact(os.path.join(results_dir, "sound_top_img_comp_e" + str(epoch) + '.png'),
                            name="sound_top_image_comp_e" + str(epoch) + '.png')




    @ex.capture
    def _record_checkpoint(self, info, _config):
        test_info = info['test']
        loss = test_info['loss']
        is_best = loss < self.best_loss
        self.best_loss = min(loss, self.best_loss)

        # TODO: Find a better way to do this. _config is a
        # ReadOnlyDict defined by sacred, which is not pickleable. And
        # due to the nested config, we can't simply do dict(_config)
        pickleable_vae_config = dict(_config['vae'])
        env_config = dict(_config['pendulum'])
        env_config['name'] = 'pendulum'
        env_config['sound_normalization'] = \
            self.train_dataloader.dataset.sound_normalization()
        vae_utils.save_checkpoint(
            {
                'state_dict': self.vae.state_dict(),
                'best_loss': self.best_loss,
                'loss': test_info['loss'],
                'image_recon_loss': test_info['image_recon_loss'],
                'sound_recon_loss': test_info['sound_recon_loss'],
                'epoch': info['epoch'],
                'optimizer': info['optimizer'].state_dict(),
                'model_config': pickleable_vae_config,
                'env_config': env_config
            },
            is_best,
            folder=log_dir_path('trained_models'))

    @ex.capture
    def __call__(self, epoch_info, _config):
        self._record_train_info(epoch_info['train'])
        self._record_test_info(epoch_info['test'])
        self._record_artifacts_maybe(epoch_info)
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
    vae_config = _config['vae']
    pendulum_config = _config['pendulum']
    gpu_config = _config['gpu']

    # Set seeds
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(vae_config['seed'])
    np.random.seed(vae_config['seed'])
    random.seed(vae_config['seed'])
    torch.cuda.manual_seed(vae_config['seed'])

    # Create model
    model = vae_utils.make_multimodalvae(vae_config, pendulum_config)

    # Create trainer
    trainer = VAETrainer(model, vae_config, gpu_config['cuda'])

    # Create Dataloaders
    train_loader, test_loader = make_dataloaders(
        pendulum_config['train_dataset_samples'],
        pendulum_config['test_dataset_samples'], pendulum_config['n_stack'],
        pendulum_config['original_frequency'],
        pendulum_config['sound_velocity'], pendulum_config['sound_receivers'],
        vae_config['batch_size'])
    post_epoch_cb = PostEpochCb(model, train_loader, test_loader)

    trainer.train(vae_config['epochs'], train_loader, test_loader,
                  _config['gpu']['cuda'], post_epoch_cb, post_cb)


@ex.main
def main(_config, _run):
    os.makedirs(log_dir_path('data'), exist_ok=True)
    os.makedirs(log_dir_path('trained_models'), exist_ok=True)
    os.makedirs(log_dir_path('results'), exist_ok=True)

    train()


if __name__ == '__main__':
    ex.run_commandline()
