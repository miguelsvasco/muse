import torch
import numpy as np
import muse.scenarios.multimodal_atari.pendulum_sound_dataset as ps_dataset


class Processor(object):
    def __init__(self, vae, sound_normalization, cuda):
        self.vae = vae
        self.sound_normalization = sound_normalization
        self.device = torch.device('cuda') if cuda else torch.device('cpu')


    def preprocess_joint(self, observation):

        image, sound = observation

        # preprocess image the same way as the dataset
        image = ps_dataset.preprocess(image)

        # torch it up
        image = torch.tensor(image).float().to(self.device)

        # fix sizes
        image = image.unsqueeze(0).unsqueeze(0)


        # Sound Preprocess
        sound = np.array(sound)

        # preprocess image the same way as the dataset
        min_freq, max_freq = self.sound_normalization['frequency']
        sound[:, 0] = (sound[:, 0] - min_freq) / (max_freq - min_freq)

        min_amp, max_amp = self.sound_normalization['amplitude']
        sound[:, 1] = (sound[:, 1] - min_amp) / (max_amp - min_amp)

        # torch it up
        sound = torch.tensor(np.array(sound)).float().to(self.device)

        # fix sizes
        sound = sound.unsqueeze(0).unsqueeze(0)

        return (image, sound)

    def postprocess_joint(self, images, sounds):

        image = torch.cat(images, dim=1)
        sound = torch.cat(sounds, dim=1)

        with torch.no_grad():
            return self.vae.gen_latents(image=image, sound=sound).detach()

    def preprocess_just_image(self, observation):
        image, _ = observation

        # preprocess image the same way as the dataset
        image = ps_dataset.preprocess(image)

        # torch it up
        image = torch.tensor(image).float().to(self.device)

        # fix sizes
        image = image.unsqueeze(0).unsqueeze(0)

        return image

    def postprocess_just_image(self, images):
        image = torch.cat(images, dim=1)
        with torch.no_grad():
            return self.vae.gen_latents(image=image, sound=None).detach()

    def preprocess_just_sound(self, observation):
        _, sound = observation

        sound = np.array(sound)

        # preprocess image the same way as the dataset
        min_freq, max_freq = self.sound_normalization['frequency']
        sound[:, 0] = (sound[:, 0] - min_freq) / (max_freq - min_freq)

        min_amp, max_amp = self.sound_normalization['amplitude']
        sound[:, 1] = (sound[:, 1] - min_amp) / (max_amp - min_amp)

        # torch it up
        sound = torch.tensor(np.array(sound)).float().to(self.device)

        # fix sizes
        sound = sound.unsqueeze(0).unsqueeze(0)

        return sound

    def postprocess_just_sound(self, sounds):
        sound = torch.cat(sounds, dim=1)
        with torch.no_grad():
            return self.vae.gen_latents(image=None, sound=sound).detach()
