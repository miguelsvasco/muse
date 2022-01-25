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


class VAETrainer(object):
    def __init__(self, model, training_config, cuda):

        # Model
        self.model = model
        self.use_cuda = cuda
        if cuda:
            self.model.cuda()

        # Training hyperparameters
        self.learning_rate = training_config['learning_rate']
        self.lambda_i = training_config['lambda_image']
        self.lambda_s = training_config['lambda_sound']
        self.beta_i = training_config['beta_image']
        self.beta_s = training_config['beta_sound']
        self.gamma_i = training_config['gamma_image']
        self.gamma_s = training_config['gamma_sound']
        self.beta = training_config['beta_top']
        self.alpha_fpa = training_config['alpha_fpa']

        # Warmups
        self.wup_mod_epochs = training_config['wup_mod_epochs']
        self.wup_top_epochs = training_config['wup_top_epochs']
        self.beta_i_wup = WarmUp(epochs=self.wup_mod_epochs, value=self.beta_i)
        self.beta_s_wup = WarmUp(epochs=self.wup_mod_epochs, value=self.beta_s)
        self.beta_t_wup = WarmUp(epochs=self.wup_top_epochs, value=self.beta)

        # Optimizer
        self.optim = optim.Adam(self.model.parameters(), lr=training_config['learning_rate'])


    def loss_function(self, image_data, sound_data, image_out, sound_out, image_dist, sound_dist, mu, logvar, fpa_mu, fpa_logvar):

        # Image Bottom loss Terms
        image_recon = self.lambda_i * torch.sum(F.binary_cross_entropy(image_out.view(image_out.size(0), -1),
                                                                       image_data.view(image_data.size(0), -1),
                                                                       reduction='none'), dim=-1)
        image_prior = self.beta_i * \
                      (-0.5 * torch.sum(1 + image_dist[1] - image_dist[0].pow(2) - image_dist[1].exp(), dim=1))

        # Sound Bottom loss Terms
        sound_recon = self.lambda_s * torch.sum(F.mse_loss(sound_out.view(sound_out.size(0), -1),
                                                           sound_data.view(sound_data.size(0), -1),
                                                           reduction='none'), dim=-1)
        sound_prior = self.beta_s * \
                      (-0.5 * torch.sum(1 + sound_dist[1] - sound_dist[0].pow(2) - sound_dist[1].exp(), dim=1))

        # Top Representation Terms
        mnist_top_recon = self.gamma_i * torch.sum(F.mse_loss(image_dist[3], image_dist[2].clone().detach(),
                                                              reduction='none'), dim=-1)
        sound_top_recon = self.gamma_s * torch.sum(F.mse_loss(sound_dist[3], sound_dist[2].clone().detach(),
                                                              reduction='none'), dim=-1)

        top_prior = self.beta_t_wup.get() * \
                    (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        top_fpa = 0.0
        for i in range(len(fpa_mu)):
            top_fpa += self.alpha_fpa * sym_KLD_gaussian(q_mu=mu, q_logvar=logvar, p_mu=fpa_mu[i],
                                                         p_logvar=fpa_logvar[i])
        top_fpa /= len(fpa_mu)

        # Total loss
        loss = torch.mean(image_recon + sound_recon + image_prior + sound_prior + mnist_top_recon + sound_top_recon + top_prior + top_fpa)

        return loss, image_recon, sound_recon, image_prior, sound_prior, mnist_top_recon, sound_top_recon, top_prior, top_fpa


    def _run(self, train, epoch, dataloader, cuda):

        if train:
            str_name = 'Train'
            self.model.train()
        else:
            str_name = 'Eval'
            self.model.eval()

        # Meters
        loss_meter = AverageMeter()
        image_recon_meter = AverageMeter()
        image_prior_meter = AverageMeter()
        sound_recon_meter = AverageMeter()
        sound_prior_meter = AverageMeter()
        image_top_recon_meter = AverageMeter()
        sound_top_recon_meter = AverageMeter()
        prior_meter = AverageMeter()
        fpa_meter = AverageMeter()

        for batch_idx, (image, sound) in enumerate(dataloader):

            if cuda:
                image, sound = image.cuda(), sound.cuda()

            if train:
                self.optim.zero_grad()

            # Forward
            image_out, sound_out, \
            image_dist, sound_dist, \
            mu, logvar, fpa_mu, fpa_logvar = self.model(image, sound)

            # Losses
            loss, image_recon, sound_recon, image_prior, sound_prior,\
            image_top_recon, sound_top_recon,\
            prior, fpa = self.loss_function(image, sound, image_out, sound_out, image_dist, sound_dist,
                                                          mu, logvar, fpa_mu, fpa_logvar)

            if train:
                loss.backward()
                self.optim.step()

            # Update meters
            loss_meter.update(loss.item(), len(image))
            image_recon_meter.update(torch.mean(image_recon).item(), len(image))
            image_prior_meter.update(torch.mean(image_prior).item(), len(image))
            sound_recon_meter.update(torch.mean(sound_recon).item(), len(image))
            sound_prior_meter.update(torch.mean(sound_prior).item(), len(image))
            image_top_recon_meter.update(torch.mean(image_top_recon).item(), len(image))
            sound_top_recon_meter.update(torch.mean(sound_top_recon).item(), len(image))
            prior_meter.update(torch.mean(prior).item(), len(image))
            fpa_meter.update(torch.mean(fpa).item(), len(image))


            # log every 100 batches
            if batch_idx % 100 == 0:
                print(f'{str_name} Epoch: {epoch} '
                      f'({100. * batch_idx / len(dataloader):.0f}%)]\t'
                      f'Loss: {loss_meter.avg:.6f}')

        print(f'====> Epoch: {epoch}\t' f'Loss: {loss_meter.avg:.4f}')

        info = {
            'loss': loss_meter.avg,
            'image_recon_loss': image_recon_meter.avg,
            'image_prior_loss': image_prior_meter.avg,
            'sound_recon_loss': sound_recon_meter.avg,
            'sound_prior_loss': sound_prior_meter.avg,
            'image_top_recon_loss': image_top_recon_meter.avg,
            'sound_top_recon_loss': sound_top_recon_meter.avg,
            'prior_loss': prior_meter.avg,
            'fpa_loss': fpa_meter.avg,
        }
        return info

    def train(self,
               epochs,
               train_dataloader,
               test_dataloader,
               cuda,
               post_epoch_cb=None,
               post_cb=None):

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
            self.beta_i_wup.update()
            self.beta_s_wup.update()
            self.beta_t_wup.update()

            is_training = True
            train_info = self._run(is_training, epoch, train_dataloader,
                                   cuda)
            is_training = False
            test_info = self._run(is_training, epoch, test_dataloader, cuda)

            info['model'] = self.model
            info['optimizer'] = self.optim
            info['epoch'] = epoch
            info['train'] = train_info
            info['test'] = test_info
            post_epoch_cb(info)

        post_cb(info)