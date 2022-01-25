import os
import sys
import sacred
import random
import torch
import torchvision
import numpy as np
import torch.nn.functional as F
from muse.evaluation.standard.mnist_svhn.model.model import MUSE
import muse.evaluation.standard.mnist_svhn.ingredients as ingredients
from muse.evaluation.standard.mnist_svhn.model.trainer import Trainer
from muse.scenarios.standard_dataset.double_dataset import DoubleDataset
import muse.evaluation.standard.mnist_svhn.model.training_utils as t_utils

ex = sacred.Experiment(
    'mnist_svhn_muse_train',
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
        _run.log_scalar('train_mnist_recon_loss', info['mnist_recon_loss'])
        _run.log_scalar('train_mnist_prior_loss', info['mnist_prior_loss'])
        _run.log_scalar('train_svhn_recon_loss', info['svhn_recon_loss'])
        _run.log_scalar('train_svhn_prior_loss', info['svhn_prior_loss'])
        _run.log_scalar('train_label_recon_loss', info['label_recon_loss'])
        _run.log_scalar('train_label_prior_loss', info['label_prior_loss'])
        _run.log_scalar('train_mnist_top_recon_loss', info['mnist_top_recon_loss'])
        _run.log_scalar('train_svhn_top_recon_loss', info['svhn_top_recon_loss'])
        _run.log_scalar('train_label_top_recon_loss', info['label_top_recon_loss'])
        _run.log_scalar('train_top_prior_loss', info['top_prior_loss'])
        _run.log_scalar('train_top_fpa_loss', info['top_fpa_loss'])

    @ex.capture
    def _record_test_info(self, info, _run):
        _run.log_scalar('test_loss', info['loss'])
        _run.log_scalar('test_mnist_recon_loss', info['mnist_recon_loss'])
        _run.log_scalar('test_mnist_prior_loss', info['mnist_prior_loss'])
        _run.log_scalar('test_svhn_recon_loss', info['svhn_recon_loss'])
        _run.log_scalar('test_svhn_prior_loss', info['svhn_prior_loss'])
        _run.log_scalar('test_label_recon_loss', info['label_recon_loss'])
        _run.log_scalar('test_label_prior_loss', info['label_prior_loss'])
        _run.log_scalar('test_mnist_top_recon_loss', info['mnist_top_recon_loss'])
        _run.log_scalar('test_svhn_top_recon_loss', info['svhn_top_recon_loss'])
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
                mnist_data = test_data[1]
                svhn_data = test_data[2]
                label_data = torch.nn.functional.one_hot(test_data[0], num_classes=10).float()

                if self.model.use_cuda:
                    mnist_data = mnist_data.cuda()
                    svhn_data = svhn_data.cuda()
                    label_data = label_data.cuda()

                    # Generate modalities
                    mnist_out, svhn_out, sym_out = self.model.generate(x_mnist=mnist_data,
                                                                       x_svhn=svhn_data,
                                                                       x_label=label_data)
                    m_out, m_top_out = mnist_out[0], mnist_out[1]
                    s_out, s_top_out = svhn_out[0], svhn_out[1]
                    l_out, l_top_out = sym_out[0], sym_out[1]

                    m_m_out, m_s_out, m_l_out = self.model.generate(x_mnist=mnist_data)
                    m_m_out, m_s_out, m_l_out = m_m_out[1], m_s_out[1], m_l_out[1]

                    s_m_out, s_s_out, s_l_out = self.model.generate(x_svhn=svhn_data)
                    s_m_out, s_s_out, s_l_out =  s_m_out[1], s_s_out[1], s_l_out[1]

                    l_m_out, l_s_out, l_l_out = self.model.generate(x_label=label_data)
                    l_m_out, l_s_out, l_l_out = l_m_out[1], l_s_out[1], l_l_out[1]

                    ms_m_out, ms_s_out, ms_l_out = self.model.generate(x_mnist=mnist_data, x_svhn=svhn_data)
                    ms_m_out, ms_s_out, ms_l_out =  ms_m_out[1], ms_s_out[1], ms_l_out[1]

                    ml_m_out, ml_s_out, ml_l_out = self.model.generate(x_mnist=mnist_data, x_label=label_data)
                    ml_m_out, ml_s_out, ml_l_out =  ml_m_out[1], ml_s_out[1], ml_l_out[1]

                    sl_m_out, sl_s_out, sl_l_out = self.model.generate(x_svhn=svhn_data, x_label=label_data)
                    sl_m_out, sl_s_out, sl_l_out = sl_m_out[1], sl_s_out[1], sl_l_out[1]

                    # MNIST Recon
                    mnist_comp = torch.cat([mnist_data.view(-1, 1, 28, 28).cpu(), m_out.view(-1, 1, 28, 28).cpu()])
                    mnist_top_comp = torch.cat([mnist_data.view(-1, 1, 28, 28).cpu(), m_top_out.view(-1, 1, 28, 28).cpu()])
                    m_z_m_comp = torch.cat([mnist_data.view(-1, 1, 28, 28).cpu(), m_m_out.view(-1, 1, 28, 28).cpu()])
                    s_z_m_comp = torch.cat([mnist_data.view(-1, 1, 28, 28).cpu(), s_m_out.view(-1, 1, 28, 28).cpu()])
                    l_z_m_comp = torch.cat([mnist_data.view(-1, 1, 28, 28).cpu(), l_m_out.view(-1, 1, 28, 28).cpu()])

                    ms_z_m_comp = torch.cat([mnist_data.view(-1, 1, 28, 28).cpu(), ms_m_out.view(-1, 1, 28, 28).cpu()])
                    ml_z_m_comp = torch.cat([mnist_data.view(-1, 1, 28, 28).cpu(), ml_m_out.view(-1, 1, 28, 28).cpu()])
                    sl_z_m_comp = torch.cat([mnist_data.view(-1, 1, 28, 28).cpu(), sl_m_out.view(-1, 1, 28, 28).cpu()])

                    # SVHN Recon
                    svhn_comp = torch.cat([svhn_data.view(-1, 3, 32, 32).cpu(), s_out.view(-1, 3, 32, 32).cpu()])
                    svhn_top_comp = torch.cat([svhn_data.view(-1, 3, 32, 32).cpu(), s_top_out.view(-1, 3, 32, 32).cpu()])
                    m_z_s_comp = torch.cat([svhn_data.view(-1, 3, 32, 32).cpu(), m_s_out.view(-1, 3, 32, 32).cpu()])
                    s_z_s_comp = torch.cat([svhn_data.view(-1, 3, 32, 32).cpu(), s_s_out.view(-1, 3, 32, 32).cpu()])
                    l_z_s_comp = torch.cat([svhn_data.view(-1, 3, 32, 32).cpu(), l_s_out.view(-1, 3, 32, 32).cpu()])

                    ms_z_s_comp = torch.cat([svhn_data.view(-1, 3, 32, 32).cpu(), ms_s_out.view(-1, 3, 32, 32).cpu()])
                    ml_z_s_comp = torch.cat([svhn_data.view(-1, 3, 32, 32).cpu(), ml_s_out.view(-1, 3, 32, 32).cpu()])
                    sl_z_s_comp = torch.cat([svhn_data.view(-1, 3, 32, 32).cpu(), sl_s_out.view(-1, 3, 32, 32).cpu()])

                    # Text Recon
                    sym_res = np.argmax(F.log_softmax(l_out, dim=-1).cpu().numpy(), axis=1).tolist()
                    sym_res_str = ''
                    for i, item in enumerate(sym_res):
                        sym_res_str += str(item) + " "

                    sym_top_res = np.argmax(F.log_softmax(l_top_out, dim=-1).cpu().numpy(), axis=1).tolist()
                    sym_top_res_str = ''
                    for i, item in enumerate(sym_top_res):
                        sym_top_res_str += str(item) + " "

                    m_z_l_out_res = np.argmax(F.log_softmax(m_l_out, dim=-1).cpu().numpy(), axis=1).tolist()
                    m_z_l_out_res_str = ''
                    for i, item in enumerate(m_z_l_out_res):
                        m_z_l_out_res_str += str(item) + " "

                    s_z_l_out_res = np.argmax(F.log_softmax(s_l_out, dim=-1).cpu().numpy(), axis=1).tolist()
                    s_z_l_out_res_str = ''
                    for i, item in enumerate(s_z_l_out_res):
                        s_z_l_out_res_str += str(item) + " "

                    l_z_l_out_res = np.argmax(F.log_softmax(l_l_out, dim=-1).cpu().numpy(), axis=1).tolist()
                    l_z_l_out_res_str = ''
                    for i, item in enumerate(l_z_l_out_res):
                        l_z_l_out_res_str += str(item) + " "

                    ms_z_l_out_res = np.argmax(F.log_softmax(ms_l_out, dim=-1).cpu().numpy(), axis=1).tolist()
                    ms_z_l_out_res_str = ''
                    for i, item in enumerate(ms_z_l_out_res):
                        ms_z_l_out_res_str += str(item) + " "

                    ml_z_l_out_res = np.argmax(F.log_softmax(ml_l_out, dim=-1).cpu().numpy(), axis=1).tolist()
                    ml_z_l_out_res_str = ''
                    for i, item in enumerate(ml_z_l_out_res):
                        ml_z_l_out_res_str += str(item) + " "

                    sl_z_l_out_res = np.argmax(F.log_softmax(sl_l_out, dim=-1).cpu().numpy(), axis=1).tolist()
                    sl_z_l_out_res_str = ''
                    for i, item in enumerate(sl_z_l_out_res):
                        sl_z_l_out_res_str += str(item) + " "

                    # Save data
                    # MNIST
                    torchvision.utils.save_image(torchvision.utils.make_grid(mnist_comp,
                                                                             padding=5,
                                                                             pad_value=.5,
                                                                             nrow=mnist_data.size(0)),
                                                 os.path.join(results_dir, 'mnist_mod_e' + str(epoch) + '.png'))
                    ex.add_artifact(os.path.join(results_dir, "mnist_mod_e" + str(epoch) + '.png'),
                                    name="mnist_recon_e" + str(epoch) + '.png')

                    torchvision.utils.save_image(torchvision.utils.make_grid(mnist_top_comp,
                                                                             padding=5,
                                                                             pad_value=.5,
                                                                             nrow=mnist_data.size(0)),
                                                 os.path.join(results_dir, 'mnist_top_e' + str(epoch) + '.png'))
                    ex.add_artifact(os.path.join(results_dir, "mnist_top_e" + str(epoch) + '.png'),
                                    name="mnist_top_recon_e" + str(epoch) + '.png')

                    torchvision.utils.save_image(torchvision.utils.make_grid(m_z_m_comp,
                                                                             padding=5,
                                                                             pad_value=.5,
                                                                             nrow=mnist_data.size(0)),
                                                 os.path.join(results_dir, 'mnist_z_mnist_mod_e' + str(epoch) + '.png'))
                    ex.add_artifact(os.path.join(results_dir, "mnist_z_mnist_mod_e" + str(epoch) + '.png'),
                                    name="mnist_z_mnist_recon_e" + str(epoch) + '.png')

                    torchvision.utils.save_image(torchvision.utils.make_grid(s_z_m_comp,
                                                                             padding=5,
                                                                             pad_value=.5,
                                                                             nrow=mnist_data.size(0)),
                                                 os.path.join(results_dir, 'svhn_z_mnist_mod_e' + str(epoch) + '.png'))
                    ex.add_artifact(os.path.join(results_dir, "svhn_z_mnist_mod_e" + str(epoch) + '.png'),
                                    name="svhn_z_mnist_recon_e" + str(epoch) + '.png')

                    torchvision.utils.save_image(torchvision.utils.make_grid(l_z_m_comp,
                                                                             padding=5,
                                                                             pad_value=.5,
                                                                             nrow=mnist_data.size(0)),
                                                 os.path.join(results_dir, 'label_z_mnist_mod_e' + str(epoch) + '.png'))
                    ex.add_artifact(os.path.join(results_dir, "label_z_mnist_mod_e" + str(epoch) + '.png'),
                                    name="label_z_mnist_recon_e" + str(epoch) + '.png')

                    torchvision.utils.save_image(torchvision.utils.make_grid(ms_z_m_comp,
                                                                             padding=5,
                                                                             pad_value=.5,
                                                                             nrow=mnist_data.size(0)),
                                                 os.path.join(results_dir, 'ms_z_mnist_mod_e' + str(epoch) + '.png'))
                    ex.add_artifact(os.path.join(results_dir, "ms_z_mnist_mod_e" + str(epoch) + '.png'),
                                    name="ms_z_mnist_recon_e" + str(epoch) + '.png')

                    torchvision.utils.save_image(torchvision.utils.make_grid(ml_z_m_comp,
                                                                             padding=5,
                                                                             pad_value=.5,
                                                                             nrow=mnist_data.size(0)),
                                                 os.path.join(results_dir, 'ml_z_mnist_mod_e' + str(epoch) + '.png'))
                    ex.add_artifact(os.path.join(results_dir, "ml_z_mnist_mod_e" + str(epoch) + '.png'),
                                    name="ml_z_mnist_recon_e" + str(epoch) + '.png')

                    torchvision.utils.save_image(torchvision.utils.make_grid(sl_z_m_comp,
                                                                             padding=5,
                                                                             pad_value=.5,
                                                                             nrow=mnist_data.size(0)),
                                                 os.path.join(results_dir, 'sl_z_mnist_mod_e' + str(epoch) + '.png'))
                    ex.add_artifact(os.path.join(results_dir, "sl_z_mnist_mod_e" + str(epoch) + '.png'),
                                    name="sl_z_mnist_recon_e" + str(epoch) + '.png')

                    # SVHN
                    torchvision.utils.save_image(torchvision.utils.make_grid(svhn_comp,
                                                                             padding=5,
                                                                             pad_value=.5,
                                                                             nrow=mnist_data.size(0)),
                                                 os.path.join(results_dir, 'svhn_mod_e' + str(epoch) + '.png'))
                    ex.add_artifact(os.path.join(results_dir, "svhn_mod_e" + str(epoch) + '.png'),
                                    name="svhn_recon_e" + str(epoch) + '.png')

                    torchvision.utils.save_image(torchvision.utils.make_grid(svhn_top_comp,
                                                                             padding=5,
                                                                             pad_value=.5,
                                                                             nrow=mnist_data.size(0)),
                                                 os.path.join(results_dir, 'svhn_top_e' + str(epoch) + '.png'))
                    ex.add_artifact(os.path.join(results_dir, "svhn_top_e" + str(epoch) + '.png'),
                                    name="svhn_top_recon_e" + str(epoch) + '.png')

                    torchvision.utils.save_image(torchvision.utils.make_grid(m_z_s_comp,
                                                                             padding=5,
                                                                             pad_value=.5,
                                                                             nrow=mnist_data.size(0)),
                                                 os.path.join(results_dir, 'mnist_z_svhn_mod_e' + str(epoch) + '.png'))
                    ex.add_artifact(os.path.join(results_dir, "mnist_z_svhn_mod_e" + str(epoch) + '.png'),
                                    name="mnist_z_svhn_recon_e" + str(epoch) + '.png')

                    torchvision.utils.save_image(torchvision.utils.make_grid(s_z_s_comp,
                                                                             padding=5,
                                                                             pad_value=.5,
                                                                             nrow=mnist_data.size(0)),
                                                 os.path.join(results_dir, 'svhn_z_svhn_mod_e' + str(epoch) + '.png'))
                    ex.add_artifact(os.path.join(results_dir, "svhn_z_svhn_mod_e" + str(epoch) + '.png'),
                                    name="svhn_z_svhn_recon_e" + str(epoch) + '.png')

                    torchvision.utils.save_image(torchvision.utils.make_grid(l_z_s_comp,
                                                                             padding=5,
                                                                             pad_value=.5,
                                                                             nrow=mnist_data.size(0)),
                                                 os.path.join(results_dir, 'label_z_svhn_mod_e' + str(epoch) + '.png'))
                    ex.add_artifact(os.path.join(results_dir, "label_z_svhn_mod_e" + str(epoch) + '.png'),
                                    name="label_z_svhn_recon_e" + str(epoch) + '.png')

                    torchvision.utils.save_image(torchvision.utils.make_grid(ms_z_s_comp,
                                                                             padding=5,
                                                                             pad_value=.5,
                                                                             nrow=mnist_data.size(0)),
                                                 os.path.join(results_dir, 'ms_z_svhn_mod_e' + str(epoch) + '.png'))
                    ex.add_artifact(os.path.join(results_dir, "ms_z_svhn_mod_e" + str(epoch) + '.png'),
                                    name="ms_z_svhn_recon_e" + str(epoch) + '.png')

                    torchvision.utils.save_image(torchvision.utils.make_grid(ml_z_s_comp,
                                                                             padding=5,
                                                                             pad_value=.5,
                                                                             nrow=mnist_data.size(0)),
                                                 os.path.join(results_dir, 'ml_z_svhn_mod_e' + str(epoch) + '.png'))
                    ex.add_artifact(os.path.join(results_dir, "ml_z_svhn_mod_e" + str(epoch) + '.png'),
                                    name="ml_z_svhn_recon_e" + str(epoch) + '.png')

                    torchvision.utils.save_image(torchvision.utils.make_grid(sl_z_s_comp,
                                                                             padding=5,
                                                                             pad_value=.5,
                                                                             nrow=mnist_data.size(0)),
                                                 os.path.join(results_dir, 'sl_z_svhn_mod_e' + str(epoch) + '.png'))
                    ex.add_artifact(os.path.join(results_dir, "sl_z_svhn_mod_e" + str(epoch) + '.png'),
                                    name="sl_z_svhn_recon_e" + str(epoch) + '.png')

                    with open(os.path.join(results_dir, 'label_res_str_e' + str(epoch) + '.txt'), "w") as text_file:
                        print(sym_res_str, file=text_file)
                    ex.add_artifact(os.path.join(results_dir, "label_res_str_e" + str(epoch) + '.txt'),
                                    name="label_recon_e" + str(epoch) + '.txt')

                    with open(os.path.join(results_dir, 'label_top_res_str_e' + str(epoch) + '.txt'), "w") as text_file:
                        print(sym_top_res_str, file=text_file)
                    ex.add_artifact(os.path.join(results_dir, "label_top_res_str_e" + str(epoch) + '.txt'),
                                    name="label_top_recon_e" + str(epoch) + '.txt')

                    with open(os.path.join(results_dir, 'mnist_z_label_res_str_e' + str(epoch) + '.txt'), "w") as text_file:
                        print(m_z_l_out_res_str, file=text_file)
                    ex.add_artifact(os.path.join(results_dir, "mnist_z_label_res_str_e" + str(epoch) + '.txt'),
                                    name="mnist_z_label_res_str_e" + str(epoch) + '.txt')

                    with open(os.path.join(results_dir, 'svhn_z_label_res_str_e' + str(epoch) + '.txt'), "w") as text_file:
                        print(s_z_l_out_res_str, file=text_file)
                    ex.add_artifact(os.path.join(results_dir, "svhn_z_label_res_str_e" + str(epoch) + '.txt'),
                                    name="svhn_z_label_res_str_e" + str(epoch) + '.txt')

                    with open(os.path.join(results_dir, 'label_z_label_res_str_e' + str(epoch) + '.txt'), "w") as text_file:
                        print(l_z_l_out_res_str, file=text_file)
                    ex.add_artifact(os.path.join(results_dir, "label_z_label_res_str_e" + str(epoch) + '.txt'),
                                    name="label_z_label_res_str_e" + str(epoch) + '.txt')

                    with open(os.path.join(results_dir, 'ms_z_label_res_str_e' + str(epoch) + '.txt'), "w") as text_file:
                        print(ms_z_l_out_res_str, file=text_file)
                    ex.add_artifact(os.path.join(results_dir, "ms_z_label_res_str_e" + str(epoch) + '.txt'),
                                    name="ms_z_label_res_str_e" + str(epoch) + '.txt')

                    with open(os.path.join(results_dir, 'ml_z_label_res_str_e' + str(epoch) + '.txt'), "w") as text_file:
                        print(ml_z_l_out_res_str, file=text_file)
                    ex.add_artifact(os.path.join(results_dir, "ml_z_label_res_str_e" + str(epoch) + '.txt'),
                                    name="ml_z_label_res_str_e" + str(epoch) + '.txt')

                    with open(os.path.join(results_dir, 'sl_z_label_res_str_e' + str(epoch) + '.txt'), "w") as text_file:
                        print(sl_z_l_out_res_str, file=text_file)
                    ex.add_artifact(os.path.join(results_dir, "sl_z_label_res_str_e" + str(epoch) + '.txt'),
                                    name="sl_z_label_res_str_e" + str(epoch) + '.txt')


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
    model = MUSE(mnist_latents=model_config['mnist_latent_dim'],
                  svhn_latents=model_config['svhn_latent_dim'],
                  label_latents=model_config['label_latent_dim'],
                  top_latents=model_config['top_latent_dim'],
                  use_cuda=gpu_config['cuda'])

    # Create trainer
    trainer = Trainer(model, training_config, gpu_config['cuda'])

    # Create Dataset
    dataset = DoubleDataset(
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