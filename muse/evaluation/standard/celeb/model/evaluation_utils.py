import math
import torch
import numpy as np
import torch.nn.functional as F

LOG2PI = float(np.log(2.0 * math.pi))

def unit_gaussian_log_pdf(x):
    """
    Log-likelihood of data given ~N(0, 1)
    @param x: PyTorch.Tensor
              samples from gaussian
    @return log_pdf: PyTorch.Tensor
                     log-likelihood
    """
    global LOG2PI
    log_pdf = -0.5 * LOG2PI - math.log(1.) / 2. - torch.pow(x, 2) / 2.
    return torch.sum(log_pdf, dim=1)



def log_mean_exp(x, dim=1):
    """
    log(1/k * sum(exp(x))): this normalizes x.
    @param x: PyTorch.Tensor
              samples from gaussian
    @param dim: integer (default: 1)
                which dimension to take the mean over
    @return: PyTorch.Tensor
             mean of x
    """
    m = torch.max(x, dim=dim, keepdim=True)[0]
    return m + torch.log(torch.mean(torch.exp(x - m),
                         dim=dim, keepdim=True))


def gaussian_log_pdf(x, mu, logvar):
    """
    Log-likelihood of data given ~N(mu, exp(logvar))
    @param x: samples from gaussian
    @param mu: mean of distribution
    @param logvar: log variance of distribution
    @return log_pdf: PyTorch.Tensor
                     log-likelihood
    """
    global LOG2PI
    log_pdf = -0.5 * LOG2PI - logvar / 2. - torch.pow(x - mu, 2) / (2. * torch.exp(logvar))
    return torch.sum(log_pdf, dim=1)


def categorical_log_pdf(x, mu):
    """
    Log-likelihood of data given ~Cat(mu)
    @param x: PyTorch.Tensor
              ground truth one-hot [batch_size]
    @param mu: PyTorch.Tensor
               Categorical distribution parameters
               log_softmax'd probabilities
               [batch_size x dims]
    @return log_pdf: PyTorch.Tensor
                     log-likelihood
    """
    log_pdf = torch.sum(x * mu,  dim=1)
    return log_pdf


# LOG ESTIMATES

def log_marginal_img_estimate(recon_image, image, z, mu, logvar, batch_size, n_samples):
    r"""Estimate log p(x). NOTE: this is not the objective that
    should be directly optimized.
    @param recon_image: torch.Tensor (batch size x # samples x 784)
                        reconstructed means on bernoulli
    @param image: torch.Tensor (batch size x 784)
                  original observed image
    @param z: torch.Tensor (batch_size x # samples x z dim)
              samples drawn from variational distribution
    @param mu: torch.Tensor (batch_size x # samples x z dim)
               means of variational distribution
    @param logvar: torch.Tensor (batch_size x # samples x z dim)
                   log-variance of variational distribution
    """

    log_p_x_given_z_2d = -torch.sum(F.mse_loss(input=recon_image.view(batch_size * n_samples, -1),
                                     target=image.view(batch_size * n_samples, -1),
                                     reduction='none'), dim=-1)

    log_q_z_given_x_2d = gaussian_log_pdf(z, mu, logvar)
    log_p_z_2d = unit_gaussian_log_pdf(z)

    log_weight_2d = log_p_x_given_z_2d + log_p_z_2d - log_q_z_given_x_2d
    log_weight = log_weight_2d.view(batch_size, n_samples)

    # need to compute normalization constant for weights
    # i.e. log ( mean ( exp ( log_weights ) ) )
    log_p = log_mean_exp(log_weight, dim=1)
    return torch.mean(log_p)


def log_marginal_attribute_estimate(recon_label, label, z, mu, logvar, batch_size, n_samples):

    log_p_x_given_z_2d = -torch.sum(F.binary_cross_entropy(target=label, input=recon_label, reduction='none'), dim=-1)
    log_q_z_given_x_2d = gaussian_log_pdf(z, mu, logvar)
    log_p_z_2d = unit_gaussian_log_pdf(z)

    log_weight_2d = log_p_x_given_z_2d + log_p_z_2d - log_q_z_given_x_2d
    log_weight = log_weight_2d.view(batch_size, n_samples)

    # need to compute normalization constant for weights
    # i.e. log ( mean ( exp ( log_weights ) ) )
    log_p = log_mean_exp(log_weight, dim=1)
    return torch.mean(log_p)



def log_joint_estimate(recon_image, image, recon_label, label,
                       z, mu, logvar, batch_size, n_samples):

    # Image
    log_p_x_img_given_z_img_2d = -torch.sum(F.mse_loss(input=recon_image.view(batch_size * n_samples, -1),
                                     target=image.view(batch_size * n_samples, -1),
                                     reduction='none'), dim=-1)


    # Label
    log_p_x_label_given_z_label_2d = torch.sum(F.binary_cross_entropy(target=label, input=recon_label, reduction='none'), dim=-1)


    # Top
    log_q_multimodal = gaussian_log_pdf(z, mu, logvar)
    log_prior_multimodal = unit_gaussian_log_pdf(z)

    log_weight_2d = log_p_x_img_given_z_img_2d + log_p_x_label_given_z_label_2d  + log_prior_multimodal \
                   - log_q_multimodal

    log_weight = log_weight_2d.view(batch_size, n_samples)

    # need to compute normalization constant for weights
    # i.e. log ( mean ( exp ( log_weights ) ) )
    log_p = log_mean_exp(log_weight, dim=1)

    return torch.mean(log_p)


def log_joint_estimate_seperate(recon_image, image, recon_label, label,
                       z_image, mu_image, logvar_image,
                                z_label, mu_label, logvar_label,
                                batch_size, n_samples):

    # Image
    log_p_x_img_given_z_img_2d = -torch.sum(F.mse_loss(input=recon_image.view(batch_size * n_samples, -1),
                                     target=image.view(batch_size * n_samples, -1),
                                     reduction='none'), dim=-1)

    log_q_image = gaussian_log_pdf(z_image, mu_image, logvar_image)
    log_prior_image = unit_gaussian_log_pdf(z_image)


    # Label
    log_p_x_label_given_z_label_2d = torch.sum(F.binary_cross_entropy(target=label, input=recon_label, reduction='none'), dim=-1)

    log_q_label = gaussian_log_pdf(z_label, mu_label, logvar_label)
    log_prior_label = unit_gaussian_log_pdf(z_label)

    log_weight_2d = log_p_x_img_given_z_img_2d + log_p_x_label_given_z_label_2d  + log_prior_image + log_prior_label \
                   - log_q_image - log_q_label

    log_weight = log_weight_2d.view(batch_size, n_samples)

    # need to compute normalization constant for weights
    # i.e. log ( mean ( exp ( log_weights ) ) )
    log_p = log_mean_exp(log_weight, dim=1)

    return torch.mean(log_p)



def log_conditional_image_estimate(recon_image, image, z, mu, logvar, batch_size, n_samples):

    log_p_x_given_z_2d = -torch.sum(F.mse_loss(input=recon_image.view(batch_size, n_samples, -1),
                                     target=image.view(batch_size, n_samples, -1),
                                     reduction='none'), dim=-1)
    log_q_z_given_x_2d = gaussian_log_pdf(z, mu, logvar)
    log_p_z_2d = unit_gaussian_log_pdf(z)

    log_weight_2d = log_p_x_given_z_2d + log_p_z_2d - log_q_z_given_x_2d
    log_weight = log_weight_2d.view(batch_size, n_samples)

    # need to compute normalization constant for weights
    # i.e. log ( mean ( exp ( log_weights ) ) )
    log_p = log_mean_exp(log_weight, dim=1)
    return torch.mean(log_p)


def log_conditional_attribute_estimate(recon_label, label, z, mu, logvar, batch_size, n_samples):

    log_p_x_given_z_2d = torch.sum(F.binary_cross_entropy(target=label, input=recon_label, reduction='none'), dim=-1)
    log_q_z_given_x_2d = gaussian_log_pdf(z, mu, logvar)
    log_p_z_2d = unit_gaussian_log_pdf(z)

    log_weight_2d = log_p_x_given_z_2d + log_p_z_2d - log_q_z_given_x_2d
    log_weight = log_weight_2d.view(batch_size, n_samples)

    # need to compute normalization constant for weights
    # i.e. log ( mean ( exp ( log_weights ) ) )
    log_p = log_mean_exp(log_weight, dim=1)
    return torch.mean(log_p)






