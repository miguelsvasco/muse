from muse.evaluation.atari.hyperhot.vae.model_components import *


class MUSE(nn.Module):
    def __init__(self, top_latents, image_latents, sound_latents, use_cuda=False):

        super(MUSE, self).__init__()

        # Structure
        self.use_cuda = use_cuda
        self.n_latents = top_latents
        self.image_latents = image_latents
        self.sound_latents = sound_latents

        # MNIST
        self.image_vae = VAE(latent_dim=image_latents, mod='image', use_cuda=use_cuda)
        self.sound_vae = VAE(latent_dim=sound_latents, mod='sound', use_cuda=use_cuda)

        # Top Representation
        self.image_top_encoder = TopEncoder(input_dim=image_latents, latent_dim=top_latents)
        self.sound_top_encoder = TopEncoder(input_dim=sound_latents, latent_dim=top_latents)

        self.image_top_decoder = TopDecoder(latent_dim=top_latents, out_dim=image_latents)
        self.sound_top_decoder = TopDecoder(latent_dim=top_latents, out_dim=sound_latents)

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

    def infer(self, z_image=None, z_sound=None):

        batch_size = z_image.size(0) if z_image is not None else z_sound.size(0)

        use_cuda = next(self.parameters()).is_cuda  # check if CUDA
        # initialize the universal prior expert
        mu, logvar = prior_expert((1, batch_size, self.n_latents),
                                  use_cuda=use_cuda)
        if z_image is not None:
            image_mu, image_logvar = self.image_top_encoder(z_image)
            mu = torch.cat((mu, image_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, image_logvar.unsqueeze(0)), dim=0)

        if z_sound is not None:
            sound_mu, sound_logvar = self.sound_top_encoder(z_sound)
            mu = torch.cat((mu, sound_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, sound_logvar.unsqueeze(0)), dim=0)

        # product of experts to combine gaussians
        mu, logvar = self.poe(mu, logvar)
        return mu, logvar


    def generate(self, image=None, sound=None):
        with torch.no_grad():

            # Encode MNIST Modality Data
            if image is not None:
                image_out, _, image_mu, _ = self.image_vae(image)
            else:
                image_out, image_mu = None, None

            # Encode Label Modality Data
            if sound is not None:
                sound_out, _, sound_mu, _ = self.sound_vae(sound)
            else:
                sound_out, sound_mu = None, None

            # Encode top representation
            top_mu, top_logvar = self.infer(z_image=image_mu, z_sound=sound_mu)
            top_z = self.reparametrize(top_mu, top_logvar)

            # Decode top representation
            image_z = self.image_top_decoder(top_z)
            image_top_out = self.image_vae.decoder(image_z)

            sound_z = self.sound_top_decoder(top_z)
            sound_top_out = self.sound_vae.decoder(sound_z)

        return [image_out, image_top_out], [sound_out, sound_top_out]

    def gen_latent(self, image=None, sound=None):
        with torch.no_grad():

            # Encode MNIST Modality Data
            if image is not None:
                _, _, image_mu, _ = self.image_vae(image)
            else:
                image_mu = None

            # Encode Label Modality Data
            if sound is not None:
                _, _, sound_mu, _ = self.sound_vae(sound)
            else:
                sound_mu = None

            # Encode top representation
            top_mu, top_logvar = self.infer(z_image=image_mu, z_sound=sound_mu)

        return top_mu

    def encode_latent(self, image=None, sound=None):
        with torch.no_grad():

            # Encode MNIST Modality Data
            if image is not None:
                _, _, image_mu, _ = self.image_vae(image)
            else:
                image_mu = None

            # Encode Label Modality Data
            if sound is not None:
                _, _, sound_mu, _ = self.sound_vae(sound)
            else:
                sound_mu = None

            # Encode top representation
            top_mu, top_logvar = self.infer(z_image=image_mu, z_sound=sound_mu)
            top_z = self.reparametrize(top_mu, top_logvar)

        return top_z, top_mu, top_logvar

    def forward(self, x_image, x_sound):

        # Encode Modality Data
        image_out, image_z, image_mu, image_logvar = self.image_vae(x_image)
        sound_out, sound_z, sound_mu, sound_logvar = self.sound_vae(x_sound)

        # Infer top representation
        top_mu, top_logvar = self.infer(z_image=image_z.clone().detach(), z_sound=sound_z.clone().detach())
        top_z = self.reparametrize(mu=top_mu, logvar=top_logvar)

        # FPA top representation
        fpa_mu_img, fpa_logvar_img = self.infer(z_image=image_z.clone().detach(), z_sound=None)
        fpa_mu_snd, fpa_logvar_snd = self.infer(z_image=None, z_sound=sound_z.clone().detach())

        # Decode information from top
        image_top_z = self.image_top_decoder(top_z)
        sound_top_z = self.sound_top_decoder(top_z)

        return image_out, sound_out, \
               [image_mu, image_logvar, image_z, image_top_z], \
               [sound_mu, sound_logvar, sound_z, sound_top_z], \
               top_mu, top_logvar, [fpa_mu_img, fpa_mu_snd], [fpa_logvar_img, fpa_logvar_snd]


class VAE(nn.Module):
    def __init__(self, latent_dim, mod, use_cuda=False):

        super(VAE, self).__init__()

        # Parameters
        self.latent_dim = latent_dim
        self.mod = mod
        self.use_cuda = use_cuda

        if self.mod == 'image':
            self.encoder = ImageEncoder(latent_dim=latent_dim)

            self.decoder = ImageDecoder(latent_dim=latent_dim)

        elif self.mod == 'sound':
            self.encoder = SoundEncoder(latent_dim=latent_dim)

            self.decoder = SoundDecoder(latent_dim=latent_dim)
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