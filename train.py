import torch
import torch.nn as nn
from torch import optim
from model import Audio2Keypoint
from dataset import VoxKP
from utils import KeyPointsRegLoss, keypoints_to_train

data_loader =

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Audio2Keypoint(args).to(device)

Keypts_regloss = KeyPointsRegLoss(args.reg_loss_type, args.d_input, args.lambda_motion_reg_loss)
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
        audio, real_pose = [x.to(device) for x in data]
        img_enc_piv, fake_pose, real_enc, fake_enc, real_pose_score, fake_pose_score, fake_pose_score_detached = model(audio, real_pose)

        train_real_pose = keypoints_to_train(real_pose, model.keypoints)
        train_real_pose = get_sample_output_by_config(train_real_pose, cfg)
        train_fake_pose = keypoints_to_train(fake_pose, model.keypoints)

        # Regression loss on motion or pose
        pose_regloss = Keypts_regloss(train_real_pose, train_fake_pose)

        # Compute discriminator loss and update D
        D_loss_real = Disc_loss(real_pose_score.new_ones(), real_pose_score)
        D_loss_fake = Disc_loss(fake_pose_score_detached.new_zeros(), fake_pose_score_detached)
        D_loss = D_loss_real + args.lambda_d * D_loss_fake
        D_optim.zero_grad()
        D_loss.backward()
        D_optim.step()

        # Loss for training the generator from the global D - have I fooled the global D?
        D_loss_fake = Disc_loss(fake_pose_score.new_ones(), fake_pose_score)

        # Compute encoder loss and update E
        E_enc_loss = Enc_loss(img_enc_piv, real_enc, fake_enc)
        E_loss = pose_regloss + D_loss_fake + E_enc_loss
        E_optim.zero_grad()
        E_loss.backward()
        E_optim.step()

        # Compute generator loss and update G
        G_enc_loss = G_Enc_loss(img_enc_piv, fake_enc)
        G_loss = pose_regloss + (args.lambda_gan * D_loss_fake) + (args.lambda_enc * G_enc_loss)
        G_optim.zero_grad()
        G_loss.backward()
        G_optim.step()

        iteration += 1
        print(f'Iter {iteration} (epoch {epoch}/{epochs}) : Disc loss : {D_loss}, Enc loss : {E_loss}, Gen loss : {G_loss}')
