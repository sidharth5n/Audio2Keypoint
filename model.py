import torch
import torch.nn as nn

from layers import ImageEncoderPIV, Generator, Discriminator
from utils import keypoints_to_train, get_training_keypoints
from consts import FRAMES_PER_SAMPLE, NUM_KEYPOINTS

class Audio2Keypoint(nn.Module):

    def __init__(self, args):
        super(Audio2Keypoint, self).__init__()
        self.encoder = ImageEncoderPIV(NUM_KEYPOINTS)
        self.generator = Generator(FRAMES_PER_SAMPLE, NUM_KEYPOINTS)
        self.discriminator = Discriminator(args.d_input, NUM_KEYPOINTS)
        self.register_buffer('keypoints', get_training_keypoints()) # get full body keypoints
        self.__init_weights()

    def forward(self, *args, **kwargs):
        mode = kwargs.get('mode', 'train_D')
        if 'mode' in kwargs:
            del kwargs['mode']
        return getattr(self, '_' + mode)(*args, **kwargs)

    def _train_D(self, audio_spect, real_pose):
        """
        Forward pass for training discriminator.

        Parameters
        ----------
        audio_spect     : torch.tensor of shape (B, 1, 418, 64)
                          Mel spectrogram of audio
        real_pose       : torch.tensor of shape (B, 136, 64)
                          Ground truth pose

        Returns
        -------
        real_pose_score : torch.tensor of shape (B, 16)
                          Realism score assigned by discriminator for real pose
        fake_pose_score : torch.tensor of shape (B, 16)
                          Realism score assigned by discriminator for fake pose
        """
        with torch.no_grad():
            image = real_pose[...,0].unsqueeze(2) #(B, 136, 1)
            img_enc_piv = self.encoder(image) #(B,32,1)
            fake_pose = self.generator(audio_spect, image, img_enc_piv) #(B,136,64)
        D_real_pose = keypoints_to_train(real_pose, self.keypoints) #(B,134,64)
        real_pose_score = self.discriminator(D_real_pose) #(B,16)
        D_fake_pose = keypoints_to_train(fake_pose, self.keypoints)
        fake_pose_score = self.discriminator(D_fake_pose)
        return real_pose_score, fake_pose_score

    def _train_G(self, audio_spect, real_pose):
        """
        Forward pass for training generator.

        Parameters
        ----------
        audio_spect     : torch.tensor of shape (B, 1, 418, 64)
                          Mel spectrogram of audio
        real_pose       : torch.tensor of shape (B, 136, 64)
                          Ground truth pose

        Returns
        -------
        img_enc_piv     : torch.tensor of shape (B, 32)
                          PIV encoding of input image
        fake_pose       : torch.tensor of shape (B, 136, 64)
                          Pose created by generator
        real_enc        : torch.tensor of shape (B, 32)
                          PIV encoding of real pose
        fake_enc        : torch.tensor of shape (B, 32)
                          PIV encoding of fake pose
        fake_pose_score : torch.tensor of shape (B, 16)
                          Realism score assigned by discriminator
        """
        image = real_pose[...,0].unsqueeze(2) #(B, 136, 1)
        with torch.no_grad():
            img_enc_piv = self.encoder(image) #(B,32)
        fake_pose = self.generator(audio_spect, image, img_enc_piv) #(B,136,64)
        real_enc = self.encoder(real_pose) #(B,32)
        fake_enc = self.encoder(fake_pose) #(B,32)
        D_fake_pose = keypoints_to_train(fake_pose, self.keypoints)
        fake_pose_score = self.discriminator(D_fake_pose)
        return img_enc_piv, fake_pose, real_enc, fake_enc, fake_pose_score

    def _train_E(self, audio_spect, real_pose):
        """
        Forward pass for training PIV Encoder.

        Parameters
        ----------
        audio_spect     : torch.tensor of shape (B, 1, 418, 64)
                          Mel spectrogram of audio
        real_pose       : torch.tensor of shape (B, 136, 64)
                          Ground truth pose

        Returns
        -------
        img_enc_piv     : torch.tensor of shape (B, 32)
                          PIV encoding of input image
        fake_pose       : torch.tensor of shape (B, 136, 64)
                          Pose created by generator
        real_enc        : torch.tensor of shape (B, 32)
                          PIV encoding of real pose
        fake_enc        : torch.tensor of shape (B, 32)
                          PIV encoding of fake pose
        fake_pose_score : torch.tensor of shape (B, 16)
                          Realism score assigned by discriminator
        """
        image = real_pose[...,0].unsqueeze(2) #(B, 136, 1)
        img_enc_piv = self.encoder(image) #(B,32)
        fake_pose = self.generator(audio_spect, image, img_enc_piv) #(B,136,64)
        real_enc = self.encoder(real_pose) #(B,32)
        fake_enc = self.encoder(fake_pose) #(B,32)
        D_fake_pose = keypoints_to_train(fake_pose, self.keypoints)
        fake_pose_score = self.discriminator(D_fake_pose)
        return img_enc_piv, fake_pose, real_enc, fake_enc, fake_pose_score

    @torch.no_grad()
    def _predict(self, audio_spect, real_pose):
        """
        Forward pass for validation.

        Parameters
        ----------
        audio_spect     : torch.tensor of shape (B, 1, 418, 64)
                          Mel spectrogram of audio
        real_pose       : torch.tensor of shape (B, 136, 64)
                          Ground truth pose

        Returns
        -------
        fake_pose       : torch.tensor of shape (B, 136, 64)
                          Pose created by generator
        fake_pose_score : torch.tensor of shape (B, 16)
                          Realism score assigned by discriminator
        fake_enc        : torch.tensor of shape (B, 32)
                          PIV encoding of fake pose
        img_enc_piv     : torch.tensor of shape (B, 32)
                          PIV encoding of input image
        """
        image = real_pose[...,0].unsqueeze(2) #(B, 136, 1)
        img_enc_piv = self.encoder(image) #(B,32)
        fake_pose = self.generator(audio_spect, image, img_enc_piv) #(B,136,64)
        fake_enc = self.encoder(fake_pose) #(B,32)
        D_fake_pose = keypoints_to_train(fake_pose, self.keypoints)
        fake_pose_score = self.discriminator(D_fake_pose)
        return fake_pose, fake_pose_score, fake_enc, img_enc_piv

    @torch.no_grad()
    def _sample(self, audio_spect, real_pose):
        """
        Forward pass for generating a sample.

        Parameters
        ----------
        audio_spect : torch.tensor of shape (B, 1, 418, 64)
                      Mel spectrogram of audio
        real_pose   : torch.tensor of shape (B, 136, 1)
                      Ground truth pose

        Returns
        -------
        fake_pose   : torch.tensor of shape (B, 136, 1)
                      Pose created by generator
        """
        image = real_pose[...,0].unsqueeze(2) #(B, 136, 1)
        img_enc_piv = self.encoder(image) #(B,32)
        fake_pose = self.generator(audio_spect, image, img_enc_piv) #(B,136,1)
        return fake_pose

    def __init_weights(self):
        """
        Glorot initialization of weights.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
