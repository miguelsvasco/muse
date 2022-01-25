import torch
import torch.optim as optim
import torch.nn.functional as F
from muse.utils.utils import AverageMeter, WarmUp

def sym_KLD_gaussian(p_mu, p_logvar, q_mu, q_logvar):
    """
    KL (p(x) || q(x))
    :param p_mu
    :param p_logvar
    :param q_mu
    :param q_logvar
    :return:
    """
    p_var = torch.exp(p_logvar)
    q_var = torch.exp(q_logvar)

    mu_diff = torch.pow(p_mu - q_mu, 2)
    first_term = 0.5 * (mu_diff + q_var) / p_var
    second_term = 0.5 * (mu_diff + p_var) / q_var

    return torch.sum(first_term + second_term - 1, dim=-1)


class Trainer(object):
    def __init__(self, model, training_config, cuda):

        # Model
        self.model = model
        self.use_cuda = cuda
        if cuda:
            self.model.cuda()

        # Training hyperparameters
        self.learning_rate = training_config['learning_rate']
        self.lambda_m = training_config['lambda_mnist']
        self.lambda_s = training_config['lambda_svhn']
        self.lambda_l = training_config['lambda_label']
        self.beta_m = training_config['beta_mnist']
        self.beta_s = training_config['beta_svhn']
        self.beta_l = training_config['beta_label']
        self.gamma_m = training_config['gamma_mnist']
        self.gamma_s = training_config['gamma_svhn']
        self.gamma_l = training_config['gamma_label']
        self.beta_t = training_config['beta_top']
        self.alpha_fpa = training_config['alpha_fpa']

        # Warmups
        self.wup_mod_epochs = training_config['wup_mod_epochs']
        self.wup_top_epochs = training_config['wup_top_epochs']
        self.beta_m_wup = WarmUp(epochs=self.wup_mod_epochs, value=self.beta_m)
        self.beta_s_wup = WarmUp(epochs=self.wup_mod_epochs, value=self.beta_s)
        self.beta_l_wup = WarmUp(epochs=self.wup_mod_epochs, value=self.beta_l)
        self.beta_t_wup = WarmUp(epochs=self.wup_top_epochs, value=self.beta_t)

        # Optimizer
        self.optim = optim.Adam(self.model.parameters(), lr=training_config['learning_rate'])


    def loss_function(self, mnist_data, svhn_data, label_data, mnist_out, svhn_out, label_out,
                      mnist_dist, svhn_dist, label_dist, mu, logvar, fpa_mu, fpa_logvar):

        # MNIST Bottom loss Terms
        mnist_recon = self.lambda_m * torch.sum(F.mse_loss(mnist_out.view(mnist_out.size(0), -1),
                                                           mnist_data.view(mnist_data.size(0), -1),
                                                           reduction='none'), dim=-1)
        mnist_prior = self.beta_m * \
                      (-0.5 * torch.sum(1 + mnist_dist[1] - mnist_dist[0].pow(2) - mnist_dist[1].exp(), dim=1))

        # SVHN Bottom loss Terms
        svhn_recon = self.lambda_s * torch.sum(F.mse_loss(svhn_out.view(svhn_out.size(0), -1),
                                                           svhn_data.view(svhn_data.size(0), -1),
                                                           reduction='none'), dim=-1)
        svhn_prior = self.beta_s * \
                      (-0.5 * torch.sum(1 + svhn_dist[1] - svhn_dist[0].pow(2) - svhn_dist[1].exp(), dim=1))

        # Label Bottom loss Terms
        _, targets = label_data.max(dim=1)
        label_recon = self.lambda_l * F.cross_entropy(label_out, targets, reduction='none')

        label_prior = self.beta_l * \
                      (-0.5 * torch.sum(1 + label_dist[1] - label_dist[0].pow(2) - label_dist[1].exp(), dim=1))

        # Top Representation Terms
        mnist_top_recon = self.gamma_m * torch.sum(
            F.mse_loss(mnist_dist[3], mnist_dist[2].clone().detach(), reduction='none'), dim=-1)

        svhn_top_recon = self.gamma_s * torch.sum(
            F.mse_loss(svhn_dist[3], svhn_dist[2].clone().detach(), reduction='none'), dim=-1)

        label_top_recon = self.gamma_l * torch.sum(
            F.mse_loss(label_dist[3], label_dist[2].clone().detach(), reduction='none'), dim=-1)

        top_prior = self.beta_t_wup.get() * \
                    (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        top_fpa = 0.0
        for i in range(len(fpa_mu)):
            top_fpa += self.alpha_fpa * sym_KLD_gaussian(q_mu=mu, q_logvar=logvar, p_mu=fpa_mu[i],
                                                         p_logvar=fpa_logvar[i])
        top_fpa /= len(fpa_mu)

        # Total loss
        loss = torch.mean(mnist_recon + svhn_recon + label_recon + mnist_prior + svhn_prior + label_prior
                          + mnist_top_recon + svhn_top_recon + label_top_recon + top_prior + top_fpa)

        return loss, mnist_recon, svhn_recon, label_recon,\
               mnist_prior, svhn_prior, label_prior,\
               mnist_top_recon, svhn_top_recon, label_top_recon,\
               top_prior, top_fpa


    def _run(self, train, epoch, dataloader, cuda):

        if train:
            str_name = 'Train'
            self.model.train()
        else:
            str_name = 'Eval'
            self.model.eval()

        # Meters
        loss_meter = AverageMeter()
        mnist_recon_meter = AverageMeter()
        mnist_prior_meter = AverageMeter()
        svhn_recon_meter = AverageMeter()
        svhn_prior_meter = AverageMeter()
        label_recon_meter = AverageMeter()
        label_prior_meter = AverageMeter()
        mnist_top_recon_meter = AverageMeter()
        svhn_top_recon_meter = AverageMeter()
        label_top_recon_meter = AverageMeter()
        top_prior_meter = AverageMeter()
        top_fpa_meter = AverageMeter()

        for batch_idx, modality_data in enumerate(dataloader):

            label_data = torch.nn.functional.one_hot(modality_data[0], num_classes=10).float()
            mnist_data = modality_data[1]
            svhn_data = modality_data[2]
            bs = mnist_data.size(0)

            if cuda:
                mnist_data = mnist_data.cuda()
                svhn_data = svhn_data.cuda()
                label_data = label_data.cuda()

            if train:
                self.optim.zero_grad()

            # Forward
            mnist_out, svhn_out, label_out, \
            mnist_dist, svhn_dist, label_dist, \
            top_mu, top_logvar, fpa_mu, fpa_logvar = self.model(mnist_data, svhn_data, label_data)

            # Losses
            loss, mnist_recon, svhn_recon, label_recon, mnist_prior, svhn_prior, label_prior,\
            mnist_top_recon, svhn_top_recon, label_top_recon,\
            top_prior, top_fpa = self.loss_function(mnist_data, svhn_data, label_data,
                                                    mnist_out, svhn_out, label_out,
                                                    mnist_dist, svhn_dist, label_dist,
                                                    top_mu, top_logvar, fpa_mu, fpa_logvar)

            if train:
                loss.backward()
                self.optim.step()

            # Update meters
            loss_meter.update(loss.item(), bs)
            mnist_recon_meter.update(torch.mean(mnist_recon).item(), bs)
            mnist_prior_meter.update(torch.mean(mnist_prior).item(), bs)
            svhn_recon_meter.update(torch.mean(svhn_recon).item(), bs)
            svhn_prior_meter.update(torch.mean(svhn_prior).item(), bs)
            label_recon_meter.update(torch.mean(label_recon).item(), bs)
            label_prior_meter.update(torch.mean(label_prior).item(), bs)
            mnist_top_recon_meter.update(torch.mean(mnist_top_recon).item(), bs)
            svhn_top_recon_meter.update(torch.mean(svhn_top_recon).item(), bs)
            label_top_recon_meter.update(torch.mean(label_top_recon).item(), bs)
            top_prior_meter.update(torch.mean(top_prior).item(), bs)
            top_fpa_meter.update(torch.mean(top_fpa).item(), bs)


            # log every 100 batches
            if batch_idx % 100 == 0:
                print(f'{str_name} Epoch: {epoch} '
                      f'({100. * batch_idx / len(dataloader):.0f}%)]\t'
                      f'Loss: {loss_meter.avg:.6f}')

        print(f'====> Epoch: {epoch}\t' f'Loss: {loss_meter.avg:.4f}')

        info = {
            'loss': loss_meter.avg,
            'mnist_recon_loss': mnist_recon_meter.avg,
            'mnist_prior_loss': mnist_prior_meter.avg,
            'svhn_recon_loss': svhn_recon_meter.avg,
            'svhn_prior_loss': svhn_prior_meter.avg,
            'label_recon_loss': label_recon_meter.avg,
            'label_prior_loss': label_prior_meter.avg,
            'mnist_top_recon_loss': mnist_top_recon_meter.avg,
            'svhn_top_recon_loss': svhn_top_recon_meter.avg,
            'label_top_recon_loss': label_top_recon_meter.avg,
            'top_prior_loss': top_prior_meter.avg,
            'top_fpa_loss': top_fpa_meter.avg,
        }
        return info

    def train(self, epochs, dataset=None, cuda=True, post_epoch_cb=None, post_cb=None):
        # make them always callable, to avoid None checks later on
        if post_epoch_cb is None:
            post_epoch_cb = lambda x: None
        if post_cb is None:
            post_cb = lambda x: None

        if cuda:
            self.model.cuda()

        info = {}
        for epoch in range(1, epochs + 1):

            # Update Warmups
            self.beta_m_wup.update()
            self.beta_s_wup.update()
            self.beta_l_wup.update()
            self.beta_t_wup.update()

            is_training = True
            train_info = self._run(is_training, epoch, dataset.train_loader,
                                   cuda)
            is_training = False
            test_info = self._run(is_training, epoch, dataset.val_loader, cuda)

            info['model'] = self.model
            info['optimizer'] = self.optim
            info['epoch'] = epoch
            info['train'] = train_info
            info['test'] = test_info
            post_epoch_cb(info)

        post_cb(info)