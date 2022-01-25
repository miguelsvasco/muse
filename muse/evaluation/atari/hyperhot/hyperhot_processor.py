import torch
import numpy as np
import muse.scenarios.multimodal_atari.hyperhot_dataset as hh_dataset


class Processor(object):
    def __init__(self, vae, sound_normalization, cuda):
        self.vae = vae
        self.sound_normalization = sound_normalization
        self.device = torch.device('cuda') if cuda else torch.device('cpu')

    def preprocess_joint(self, observation):
        image, sound = observation

        # preprocess image the same way as the dataset
        image = hh_dataset.preprocess(image)

        # torch it up
        image = torch.tensor(image).float().to(self.device)

        # fix sizes
        image = image.unsqueeze(0).unsqueeze(0)

        # preprocess image the same way as the dataset
        min_sound = self.sound_normalization[0]
        max_sound = self.sound_normalization[1]
        sound = np.array(sound)
        sound = (sound - min_sound) / (max_sound - min_sound)

        # torch it up
        sound = torch.tensor(np.array(sound)).float().to(self.device)

        # fix sizes
        sound = sound.unsqueeze(0).unsqueeze(0)

        return (image, sound)

    def postprocess_joint(self, images, sounds):
        image = torch.cat(images, dim=1)
        sound = torch.cat(sounds, dim=1)

        with torch.no_grad():
            return self.vae.gen_latent(image=image, sound=sound).detach()


    def preprocess_just_image(self, observation):
        image, _ = observation

        # preprocess image the same way as the dataset
        image = hh_dataset.preprocess(image)

        # torch it up
        image = torch.tensor(image).float().to(self.device)

        # fix sizes
        image = image.unsqueeze(0).unsqueeze(0)

        return image

    def postprocess_just_image(self, images):
        image = torch.cat(images, dim=1)
        with torch.no_grad():
            assert not self.vae.training
            return self.vae.gen_latent(image=image, sound=None).detach()

    def preprocess_just_sound(self, observation):
        _, sound = observation

        # preprocess image the same way as the dataset
        min_sound = self.sound_normalization[0]
        max_sound = self.sound_normalization[1]
        sound = np.array(sound)
        sound = (sound - min_sound) / (max_sound - min_sound)

        # torch it up
        sound = torch.tensor(np.array(sound)).float().to(self.device)

        # fix sizes
        sound = sound.unsqueeze(0).unsqueeze(0)

        return sound

    def postprocess_just_sound(self, sounds):
        sound = torch.cat(sounds, dim=1)
        with torch.no_grad():
            assert not self.vae.training
            return self.vae.gen_latent(image=None, sound=sound).detach()

    def preprocess_eval_image(self, observation):
        image, _ = observation
        # preprocess image the same way as the dataset
        image = hh_dataset.preprocess(image).astype(np.uint8) * 255

        return image