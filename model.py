import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import ImageEncoderPIV, Generator, Discriminator
from utils import keypoints_to_train, get_training_keypoints

class Audio2Keypoint(nn.Module):

    def __init__(self, args, seq_len = 64):
        super(Audio2Keypoint, self).__init__()
        self.encoder = ImageEncoderPIV()
        self.generator = Generator()
        self.discriminator = Discriminator(args.d_input)
        self.register_buffer('keypoints', get_training_keypoints()) # get full body keypoints

    def forward(self, image, audio, real_pose):
        img_enc_piv = self.encoder(image)
        fake_pose = self.generator(audio, real_pose, image, img_enc_piv)
        if self.training:
            # TODO : Check whether these encoder lines are required during inference
            real_enc = self.encoder(real_pose)
            fake_enc = self.encoder(fake_pose)
            D_real_pose = keypoints_to_train(real_pose, self.keypoints)
            real_pose_score = self.discriminator(D_real_pose)
            D_fake_pose = keypoints_to_train(fake_pose, self.keypoints)
            fake_pose_score = self.discriminator(D_fake_pose)
            return img_enc_piv, fake_pose, real_enc, fake_enc, real_pose_score, fake_pose_score
        else:
            return fake_pose
