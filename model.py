import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import ImageEncoderPIV, Generator, Discriminator

class Audio2Keypoint(nn.Module):

    def __init__(self, seq_len = 64):
        super(Audio2Keypoint, self).__init__()
        self.encoder = ImageEncoderPIV()
        self.generator = Generator()
        self.discriminator = Discriminator()

    def forward(self, image, audio, real_pose, fake_pose):
        img_enc_piv = self.encoder(image)
        fake_pose = self.generator(audio, real_pose, image, img_enc_piv)
        real_enc = self.encoder(real_pose)
        fake_enc = self.encoder(fake_pose)
        return image_enc_piv, fake_pose, real_enc, fake_enc

    def _get_training_keypoints(self):
        training_keypoints = []
        training_keypoints.extend(CHIN_KEYPOINTS)
        training_keypoints.extend(LEFT_BROW_KEYPOINTS)
        training_keypoints.extend(RIGHT_BROW_KEYPOINTS)
        training_keypoints.extend(NOSE_KEYPOINTS)
        training_keypoints.extend(LEFT_EYE_KEYPOINTS)
        training_keypoints.extend(RIGHT_EYE_KEYPOINTS)
        training_keypoints.extend(OUTER_LIP_KEYPOINTS)
        training_keypoints.extend(INNER_LIP_KEYPOINTS)

        training_keypoints = sorted(list(set(training_keypoints)))
        return training_keypoints                                                                                     var_list=trainable_variables)
