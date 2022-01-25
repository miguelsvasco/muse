import math
import torch
import torch.nn as nn
from torch.autograd import Variable


# Constants
class Constants(object):
    eta = 1e-6
    log2 = math.log(2)
    log2pi = math.log(2 * math.pi)
    logceilc = 88  # largest cuda v s.t. exp(v) < inf
    logfloorc = -104  # smallest cuda v s.t. exp(v) > 0

# Label
class LabelEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(LabelEncoder, self).__init__()
        self.fc1   = nn.Linear(10, 64)
        self.fc2   = nn.Linear(64, 64)
        self.fc31  = nn.Linear(64, latent_dim)
        self.fc32  = nn.Linear(64, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)


class LabelDecoder(nn.Module):
    """Parametrizes p(x|z).
    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, latent_dim):
        super(LabelDecoder, self).__init__()
        self.fc1   = nn.Linear(latent_dim, 64)
        self.fc2   = nn.Linear(64, 64)
        self.fc3   = nn.Linear(64, 64)
        self.fc4   = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, z):
        h = self.relu(self.fc1(z))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        return self.fc4(h)



# MNIST
class MNISTEncoder(nn.Module):
    """Parametrizes q(z|x).
    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, latent_dim):
        super(MNISTEncoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            Swish(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            Swish())
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 512),
            Swish(),
        )

        self.latent_dim = latent_dim

        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return self.fc_mu(x), self.fc_logvar(x)


class MNISTDecoder(nn.Module):
    """Parametrizes p(x|z).
    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, latent_dim):
        super(MNISTDecoder, self).__init__()
        self.n_latents = latent_dim
        self.upsampler = nn.Sequential(
            nn.Linear(latent_dim, 512),
            Swish(),
            nn.Linear(512, 128 * 7 * 7),
            Swish())
        self.hallucinate = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            Swish(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Sigmoid())

    def forward(self, z):
        z = self.upsampler(z)
        z = z.view(-1, 128, 7, 7)
        out = self.hallucinate(z)
        return out


# SVHN
class SVHNEncoder(nn.Module):
    """ Generate latent parameters for SVHN image data. """

    def __init__(self, latent_dim):
        super(SVHNEncoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=True),
            Swish(),
            nn.Conv2d(32, 32 * 2, 4, 2, 1, bias=True),
            Swish(),
            nn.Conv2d(32 * 2, 32 * 4, 4, 2, 1, bias=True),
            Swish(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            Swish(),
            nn.Linear(1024, 512),
            Swish(),
        )

        self.latent_dim = latent_dim
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return self.fc_mu(x), self.fc_logvar(x)


class SVHNDecoder(nn.Module):
    """ Generate a SVHN image given a sample from the latent space. """

    def __init__(self, latent_dim):
        super(SVHNDecoder, self).__init__()
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 32 * 4, 4, 1, 0, bias=True),
            Swish(),
            nn.ConvTranspose2d(32 * 4, 32 * 2, 4, 2, 1, bias=True),
            Swish(),
            nn.ConvTranspose2d(32 * 2, 32, 4, 2, 1, bias=True),
            Swish(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, z):
        z = z.unsqueeze(-1).unsqueeze(-1)  # fit deconv layers
        out = self.dec(z.view(-1, *z.size()[-3:]))
        out = out.view(*z.size()[:-3], *out.size()[1:])
        return out


# Top Representation

class TopEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(TopEncoder, self).__init__()
        self.fc1   = nn.Linear(input_dim, 512)
        self.fc2   = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc31  = nn.Linear(512, latent_dim)
        self.fc32  = nn.Linear(512, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        return self.fc31(h), self.fc32(h)


class TopDecoder(nn.Module):
    """Parametrizes p(x|z).
    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, latent_dim, out_dim):
        super(TopDecoder, self).__init__()
        self.fc1   = nn.Linear(latent_dim, 512)
        self.fc2   = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4   = nn.Linear(512, out_dim)
        self.relu = nn.ReLU()

    def forward(self, z):
        h = self.relu(self.fc1(z))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        return self.fc4(h)


class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.
    @param mu: M x D for M experts
    @param logvar: M x D for M experts
    """
    def forward(self, mu, logvar, eps=1e-8):
        var       = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T         = 1. / (var + eps)
        pd_mu     = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var    = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var + eps)
        return pd_mu, pd_logvar


# Extra Components
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def prior_expert(size, use_cuda=False):
    """Universal prior expert. Here we use a spherical
    Gaussian: N(0, 1).
    @param size: integer
                 dimensionality of Gaussian
    @param use_cuda: boolean [default: False]
                     cast CUDA on variables
    """
    mu     = Variable(torch.zeros(size))
    logvar = Variable(torch.zeros(size))
    if use_cuda:
        mu, logvar = mu.cuda(), logvar.cuda()
    return mu, logvar