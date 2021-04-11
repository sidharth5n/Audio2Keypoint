import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import ImageEncoderPIV, Generator,Generator2 ,Discriminator
from utils import keypoints_to_train

class Audio2Keypoint(nn.Module):
    def __init__(self, args, seq_len = 64,training=True):
        super(Audio2Keypoint, self).__init__()
        self.encoder = ImageEncoderPIV()
        self.generator = Generator()
        self.discriminator = Discriminator(args.d_input)
        self.training=training

    def forward(self, image, audio, real_pose):
        img_enc_piv = self.encoder(image)
        # img_enc_piv.shape is 1,32,136
        fake_pose = self.generator(audio, real_pose, image, img_enc_piv)
        if self.training:
            # TODO : Check whether these encoder lines are required during inference
            real_enc = self.encoder(real_pose)
            fake_enc = self.encoder(fake_pose)
            D_training_keypoints = self._get_training_keypoints() # get full body keypoints
            D_real_pose = keypoints_to_train(real_pose, D_training_keypoints)
            real_pose_score = self.discriminator(D_real_pose)
            D_fake_pose = keypoints_to_train(fake_pose, D_training_keypoints)
            fake_pose_score = self.discriminator(D_fake_pose)
            return img_enc_piv, fake_pose, real_enc, fake_enc, real_pose_score, fake_pose_score
        else:
            return fake_pose

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
        return training_keypoints


class Audio2Keypoint2(nn.Module):
    def __init__(self, args, seq_len = 64,training=True):
        super(Audio2Keypoint, self).__init__()
        self.generator = Generator2()
        self.discriminator = Discriminator(args.d_input)
        self.training=training

    def forward(self, image, audio, real_pose):
        # img_enc_piv = self.encoder(image)
        # img_enc_piv.shape is 1,32,136
        fake_pose = self.generator(audio, real_pose, image, image)
        if self.training:
            # TODO : Check whether these encoder lines are required during inference
            # real_enc = self.encoder(real_pose)
            # fake_enc = self.encoder(fake_pose)
            D_training_keypoints = self._get_training_keypoints() # get full body keypoints
            D_real_pose = keypoints_to_train(real_pose, D_training_keypoints)
            D_real_pose = get_sample_output_by_config(D_real_pose, cfg)
            real_pose_score = self.discriminator(D_real_pose)
            D_fake_pose = keypoints_to_train(fake_pose, D_training_keypoints)
            train_real_pose=train_real_pose[:,:,0:8]
            fake_pose_score = self.discriminator(D_fake_pose)
            return D_real_pose,D_fake_pose,real_pose_score, fake_pose_score
        else:
            return fake_pose_score

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
        return training_keypoints                                                                                              
