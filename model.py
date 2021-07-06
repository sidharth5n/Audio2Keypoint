import torch
import torch.nn as nn
from layers import ImageEncoderPIV, Generator, Discriminator
from utils import keypoints_to_train, get_training_keypoints

class Audio2Keypoint(nn.Module):

    def __init__(self, args):
        super(Audio2Keypoint, self).__init__()
        self.encoder = ImageEncoderPIV()
        self.generator = Generator()
        self.discriminator = Discriminator(args.d_input)
        self.register_buffer('keypoints', get_training_keypoints()) # get full body keypoints
        self.__init_weights()

    def forward(self, audio, real_pose):
        """
        Parameters
        ----------
        audio                    : torch.tensor of shape (B, config.input_shape[1])
        real_pose                : torch.tensor of shape (B, 136, 64)

        Returns
        -------
        img_enc_piv              : torch.tensor of shape (B, 32, 1)
        fake_pose                : torch.tensor of shape (B, 136, 64)
        real_enc                 : torch.tensor of shape (B, 32, 64)
        fake_enc                 : torch.tensor of shape (B, 32, 64)
        real_pose_score          : torch.tensor of shape (B, 16)
        fake_pose_score          : torch.tensor of shape (B, 16)
        fake_pose_score_detached : torch.tensor of shape (B, 16)
        """
        image = real_pose[...,0].unsqueeze(2) #(B, 136, 1)
        img_enc_piv = self.encoder(image) #(B,32,1)
        fake_pose = self.generator(audio, real_pose, image, img_enc_piv) #(B,136,64)
        if self.training:
            # TODO : Check whether these encoder lines are required during inference
            real_enc = self.encoder(real_pose) #(B,32,64)
            fake_enc = self.encoder(fake_pose) #(B,32,64)
            D_real_pose = keypoints_to_train(real_pose, self.keypoints) #(B,134,64)
            real_pose_score = self.discriminator(D_real_pose) #(B,16)
            D_fake_pose = keypoints_to_train(fake_pose, self.keypoints)
            fake_pose_score = self.discriminator(D_fake_pose)
            D_fake_pose = keypoints_to_train(fake_pose.detach(), self.keypoints)
            fake_pose_score_detached = self.discriminator(D_fake_pose)
            return img_enc_piv, fake_pose, real_enc, fake_enc, real_pose_score, fake_pose_score, fake_pose_score_detached
        else:
            return fake_pose

    def __init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
