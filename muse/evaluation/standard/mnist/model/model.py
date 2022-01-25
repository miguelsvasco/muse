from muse.evaluation.standard.mnist.model.model_components import *


class MUSE(nn.Module):
    def __init__(self, top_latents, img_latents, label_latents, use_cuda=False):

        super(MUSE, self).__init__()

        # Structure
        self.use_cuda = use_cuda
        self.top_latents = top_latents
        self.img_latents = img_latents
        self.label_latents = label_latents

        # MNIST
        self.mnist_vae = VAE(latent_dim=img_latents, mod='mnist', use_cuda=use_cuda)
        self.label_vae = VAE(latent_dim=label_latents, mod='label', use_cuda=use_cuda)

        # Top Representation
        self.mnist_top_encoder = TopEncoder(input_dim=img_latents, latent_dim=top_latents)
        self.label_top_encoder = TopEncoder(input_dim=self.label_latents, latent_dim=top_latents)

        self.mnist_top_decoder = TopDecoder(latent_dim=top_latents, out_dim=img_latents)
        self.label_top_decoder = TopDecoder(latent_dim=top_latents, out_dim=self.label_latents)

        self.poe = ProductOfExperts()

    def reparametrize(self, mu, logvar):

        # Sample epsilon from a random gaussian with 0 mean and 1 variance
        epsilon = Variable(torch.randn(mu.size()), requires_grad=False)

        # Check if cuda is selected
        if mu.is_cuda:
            epsilon = epsilon.cuda()

        # std = exp(0.5 * log_var)
        std = logvar.mul(0.5).exp_()

        # z = std * epsilon + mu
        return mu.addcmul(std, epsilon)

    def infer(self, z_mnist=None, z_label=None):

        batch_size = z_mnist.size(0) if z_mnist is not None else z_label.size(0)

        use_cuda = next(self.parameters()).is_cuda  # check if CUDA
        # initialize the universal prior expert
        mu, logvar = prior_expert((1, batch_size, self.top_latents),
                                  use_cuda=use_cuda)
        if z_mnist is not None:
            mnist_mu, mnist_logvar = self.mnist_top_encoder(z_mnist)
            mu = torch.cat((mu, mnist_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, mnist_logvar.unsqueeze(0)), dim=0)

        if z_label is not None:
            label_mu, label_logvar = self.label_top_encoder(z_label)
            mu = torch.cat((mu, label_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, label_logvar.unsqueeze(0)), dim=0)

        # product of experts to combine gaussians
        mu, logvar = self.poe(mu, logvar)
        return mu, logvar


    def generate(self, x_mnist=None, x_label=None):
        with torch.no_grad():

            # Encode MNIST Modality Data
            if x_mnist is not None:
                mnist_out, _, mnist_mu, _ = self.mnist_vae(x_mnist)
            else:
                mnist_out, mnist_mu = None, None

            # Encode Label Modality Data
            if x_label is not None:
                label_out, _, label_mu, _ = self.label_vae(x_label)
            else:
                label_out, label_mu = None, None

            # Encode top representation
            top_mu, top_logvar = self.infer(z_mnist=mnist_mu, z_label=label_mu)
            top_z = self.reparametrize(top_mu, top_logvar)

            # Decode top representation
            mnist_z = self.mnist_top_decoder(top_z)
            mnist_top_out = self.mnist_vae.decoder(mnist_z)

            label_z = self.label_top_decoder(top_z)
            label_top_out = self.label_vae.decoder(label_z)

        return [mnist_out, mnist_top_out], [label_out, label_top_out]

    def encode_latent(self, x_mnist=None, x_label=None):
        with torch.no_grad():

            # Encode MNIST Modality Data
            if x_mnist is not None:
                _, _, mnist_mu, _ = self.mnist_vae(x_mnist)
            else:
                mnist_mu = None

            # Encode Label Modality Data
            if x_label is not None:
                _, _, label_mu, _ = self.label_vae(x_label)
            else:
                label_mu = None

            # Encode top representation
            top_mu, top_logvar = self.infer(z_mnist=mnist_mu, z_label=label_mu)
            top_z = self.reparametrize(top_mu, top_logvar)

        return top_z, top_mu, top_logvar

    def forward(self, x_mnist, x_label):

        # Encode Modality Data
        mnist_out, mnist_z, mnist_mu, mnist_logvar = self.mnist_vae(x_mnist)
        label_out, label_z, label_mu, label_logvar = self.label_vae(x_label)

        # Infer top representation
        top_mu, top_logvar = self.infer(z_mnist=mnist_z.clone().detach(), z_label=label_z.clone().detach())
        top_z = self.reparametrize(mu=top_mu, logvar=top_logvar)

        # FPA top representation
        fpa_mu_img, fpa_logvar_img = self.infer(z_mnist=mnist_z.clone().detach(), z_label=None)
        fpa_mu_lbl, fpa_logvar_lbl = self.infer(z_mnist=None, z_label=label_z.clone().detach())

        # Decode information from top
        mnist_top_z = self.mnist_top_decoder(top_z)
        label_top_z = self.label_top_decoder(top_z)

        return mnist_out, label_out, \
               [mnist_mu, mnist_logvar, mnist_z, mnist_top_z], \
               [label_mu, label_logvar, label_z, label_top_z], \
               top_mu, top_logvar, [fpa_mu_img, fpa_mu_lbl], [fpa_logvar_img, fpa_logvar_lbl]


class VAE(nn.Module):
    def __init__(self, latent_dim, mod, use_cuda=False):

        super(VAE, self).__init__()

        # Parameters
        self.latent_dim = latent_dim
        self.mod = mod
        self.use_cuda = use_cuda

        if self.mod == 'mnist':
            self.encoder = MNISTEncoder(latent_dim=latent_dim)

            self.decoder = MNISTDecoder(latent_dim=latent_dim)

        elif self.mod == 'svhn':
            self.encoder = SVHNEncoder(latent_dim=latent_dim)

            self.decoder = SVHNDecoder(latent_dim=latent_dim)

        elif self.mod == 'label':
            self.encoder = LabelEncoder(latent_dim=latent_dim)

            self.decoder = LabelDecoder(latent_dim=latent_dim)

        else:
            raise ValueError("Not implemented.")


    def reparametrize(self, mu, logvar):

        # Sample epsilon from a random gaussian with 0 mean and 1 variance
        epsilon = Variable(torch.randn(mu.size()), requires_grad=False)

        # Check if cuda is selected
        if mu.is_cuda:
            epsilon = epsilon.cuda()

        # std = exp(0.5 * log_var)
        std = logvar.mul(0.5).exp_()

        # z = std * epsilon + mu
        return mu.addcmul(std, epsilon)

    def forward(self, x, sample=True):
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        out = self.decoder(z)
        return out, z, mu, logvar