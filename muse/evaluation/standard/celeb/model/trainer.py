import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from muse.utils.utils import AverageMeter, WarmUp

def KLD_gaussian(p_mu, p_logvar, q_mu, q_logvar):
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

    p_dist = Normal(p_mu, p_var)
    q_dist = Normal(q_mu, q_var)

    return torch.sum(kl_divergence(p_dist, q_dist), dim=-1)

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
        self.lambda_m = training_config['lambda_img']
        self.lambda_l = training_config['lambda_label']
        self.beta_m = training_config['beta_img']
        self.beta_l = training_config['beta_label']
        self.gamma_m = training_config['gamma_img']
        self.gamma_l = training_config['gamma_label']
        self.beta_t = training_config['beta_top']
        self.alpha_fpa = training_config['alpha_fpa']

        # Warmups
        self.wup_mod_epochs = training_config['wup_mod_epochs']
        self.wup_top_epochs = training_config['wup_top_epochs']
        self.beta_m_wup = WarmUp(epochs=self.wup_mod_epochs, value=self.beta_m)
        self.beta_l_wup = WarmUp(epochs=self.wup_mod_epochs, value=self.beta_l)
        self.beta_t_wup = WarmUp(epochs=self.wup_top_epochs, value=self.beta_t)

        # Optimizer
        self.optim = optim.Adam(self.model.parameters(), lr=training_config['learning_rate'])


    def loss_function(self, img_data, label_data, img_out, label_out, img_dist, label_dist, mu, logvar, fpa_mu, fpa_logvar):

        # Image Bottom loss Terms
        img_recon = self.lambda_m * torch.sum(F.mse_loss(img_out.view(img_out.size(0), -1),
                                                           img_data.view(img_data.size(0), -1),
                                                           reduction='none'), dim=-1)
        img_prior = self.beta_m * \
                      (-0.5 * torch.sum(1 + img_dist[1] - img_dist[0].pow(2) - img_dist[1].exp(), dim=1))

        # Label Bottom loss Terms
        label_recon = self.lambda_l * torch.sum(F.binary_cross_entropy(label_out, label_data, reduction='none'), dim=-1)

        label_prior = self.beta_l * \
                      (-0.5 * torch.sum(1 + label_dist[1] - label_dist[0].pow(2) - label_dist[1].exp(), dim=1))

        # Top Representation Terms
        img_top_recon = self.gamma_m * torch.sum(
            F.mse_loss(img_dist[3], img_dist[2].clone().detach(), reduction='none'), dim=-1)
        label_top_recon = self.gamma_l * torch.sum(
            F.mse_loss(label_dist[3], label_dist[2].clone().detach(), reduction='none'), dim=-1)

        top_prior = self.beta_t_wup.get() * \
                    (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        top_fpa = 0.0
        for i in range(len(fpa_mu)):
            top_fpa += self.alpha_fpa * sym_KLD_gaussian(p_mu=mu, p_logvar=logvar, q_mu=fpa_mu[i],
                                                         q_logvar=fpa_logvar[i])
        top_fpa /= len(fpa_mu)

        # Total loss
        loss = torch.mean(img_recon + label_recon + img_prior + label_prior + img_top_recon + label_top_recon + top_prior + top_fpa)

        return loss, img_recon, label_recon, img_prior, label_prior, img_top_recon, label_top_recon, top_prior, top_fpa


    def _run(self, train, epoch, dataloader, cuda):

        if train:
            str_name = 'Train'
            self.model.train()
        else:
            str_name = 'Eval'
            self.model.eval()

        # Meters
        loss_meter = AverageMeter()
        img_recon_meter = AverageMeter()
        img_prior_meter = AverageMeter()
        label_recon_meter = AverageMeter()
        label_prior_meter = AverageMeter()
        img_top_recon_meter = AverageMeter()
        label_top_recon_meter = AverageMeter()
        top_prior_meter = AverageMeter()
        top_fpa_meter = AverageMeter()

        for batch_idx, modality_data in enumerate(dataloader):

            img_data = modality_data[0]
            label_data = modality_data[1].float()
            bs = img_data.size(0)

            if cuda:
                img_data = img_data.cuda()
                label_data = label_data.cuda()

            if train:
                self.optim.zero_grad()

            # Forward
            img_out, label_out, \
            img_dist, label_dist, \
            top_mu, top_logvar, fpa_mu, fpa_logvar = self.model(img_data, label_data)

            # Losses
            loss, img_recon, label_recon, img_prior, label_prior,\
            img_top_recon, label_top_recon,\
            top_prior, top_fpa = self.loss_function(img_data, label_data, img_out, label_out, img_dist, label_dist,
                                                          top_mu, top_logvar, fpa_mu, fpa_logvar)

            if train:
                loss.backward()
                self.optim.step()

            # Update meters
            loss_meter.update(loss.item(), bs)
            img_recon_meter.update(torch.mean(img_recon).item(), bs)
            img_prior_meter.update(torch.mean(img_prior).item(), bs)
            label_recon_meter.update(torch.mean(label_recon).item(), bs)
            label_prior_meter.update(torch.mean(label_prior).item(), bs)
            img_top_recon_meter.update(torch.mean(img_top_recon).item(), bs)
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
            'img_recon_loss': img_recon_meter.avg,
            'img_prior_loss': img_prior_meter.avg,
            'label_recon_loss': label_recon_meter.avg,
            'label_prior_loss': label_prior_meter.avg,
            'img_top_recon_loss': img_top_recon_meter.avg,
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