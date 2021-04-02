import torch.nn as nn
from torch import optim
from model import Audio2Keypoint
from dataset import VoxKP
from utils import KeyPointsRegLoss

data_loader =

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Audio2Keypoint().to(device)

keypts_regloss = KeyPointsRegLoss(args.reg_loss_type, args.d_input, args.lambda_motion_reg_loss)
Disc_loss = nn.MSELoss()
Enc_loss = nn.TripletMarginLoss(margin = 0.5, p = 1)
G_Enc_loss = nn.L1Loss()

D_optim = optim.Adam(model.discriminator.parameters(), lr = args.lr_d)
E_optim = optim.Adam(model.encoder.parameters(), lr = args.lr_g)
G_optim = optim.Adam(model.generator.parameters(), lr = args.lr_g)

# TODO : Checkpoint saving, resume training from checkpoint

iteration = 0

for epoch in range(start_epoch, epochs):
    for data in data_loader:
        image, audio, real_pose, fake_pose = [x.to(device) for x in data]
        # TODO : Modify forward pass, image_enc_piv is not being used
        image_enc_piv, fake_pose, real_enc, fake_enc = model(image, audio, real_pose, fake_pose)

        # remove base keypoint which is always [0,0]. Keeping it may ruin GANs training due discrete problems. etc.
        training_keypoints = self._get_training_keypoints()

        train_real_pose = keypoints_to_train(real_pose, training_keypoints)
        train_real_pose = get_sample_output_by_config(train_real_pose, cfg)
        train_fake_pose = keypoints_to_train(fake_pose, training_keypoints)

        # Regression loss on motion or pose
        pose_regloss = keypts_regloss(train_real_pose, train_fake_pose)

        # Global Discriminator and Hand Discriminator

        # get full body keypoints
        D_training_keypoints = self._get_training_keypoints()
        D_real_pose = keypoints_to_train(real_pose, D_training_keypoints)
        D_fake_pose = keypoints_to_train(fake_pose, D_training_keypoints)

        # d motion or pose
        if self.args.d_input == 'motion':
            D_fake_pose_input = to_motion_delta(D_fake_pose)
            D_real_pose_input = to_motion_delta(D_real_pose)
        elif self.args.d_input == 'pose':
            D_fake_pose_input = D_fake_pose
            D_real_pose_input = D_real_pose
        elif self.args.d_input == 'both':
            # concatenate on the temporal axis
            D_fake_pose_input = torch.cat([D_fake_pose, to_motion_delta(D_fake_pose)], dim = 1) # check dim
            D_real_pose_input = torch.cat([D_real_pose, to_motion_delta(D_real_pose)], dim = 1) # check dim

        # TODO : Move the discriminator part to model.py
        fake_pose_score = self.discriminator(D_fake_pose_input)
        real_pose_score = self.discriminator(D_real_pose_input)

        # loss for training the global D
        D_loss = Disc_loss(real_pose_score.new_ones(), real_pose_score) +
                      args.lambda_d * Disc_loss(fake_pose_score.new_zeros(), fake_pose_score)
        # train global D
        D_optim.zero_grad()
        D_loss.backward()
        D_optim.step()

        # Loss for training the generator from the global D - have I fooled the global D?
        G_gan_loss = Disc_loss(fake_pose_score.new_ones(), fake_pose_score)

        # encoder loss and train encoder
        E_enc_loss = Enc_loss(img_enc_piv, real_enc, fake_enc)
        E_loss = pose_regloss + G_gan_loss + E_enc_loss
        E_optim.zero_grad()
        E_loss.backward()
        E_optim.step()

        # loss for generatar encoding
        G_enc_loss = G_Enc_loss(img_enc_piv, fake_enc)
        # sum up ALL the losses for training the generator
        G_loss = pose_regloss + (args.lambda_gan * G_gan_loss) + (args.lambda_enc * G_enc_loss)
        G_optim.zero_grad()
        G_loss.backward()
        G_optim.step()

        iteration += 1
        print(f'Iter {iteration} (epoch {epoch}/{epochs}) : Disc loss : {D_loss}, Enc loss : {E_loss}, Gen loss : {G_loss}')
