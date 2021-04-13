from config import create_parser
import argparse
from dataload import a2kData
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
# from model import Audio2Keypoint

# Add commented batchNorm
from layers import Discriminator, Generator, ImageEncoderPIV, ImageEncoderPV
from utils import MelSpectrogram
# from dataset import VoxKP
from utils import KeyPointsRegLoss, keypoints_to_train, _get_training_keypoints
from config import get_config
from common.pose_logic_lib import translate_keypoints, get_sample_output_by_config
AUDIO_SHAPE = 67267
my_dict = {"audio_fn": ["./Gestures/human/train/audio/id0004462OEFEevKvs00001-00:00:00.040000-00:00:05.360000.wav"],
           "dataset": ["train"], "end": ["0:00:02.560000"], "interval_id": ["id037619diQL48epnM00019"],
           "pose_fn": ["./Gestures/human/train/npz/id0004.npz"],
           "speaker": ["human"], "start": ["0:00:00.040000"], "video_fn": ["id03761/9diQL48epnM/00019.mp4"]}
df = pd.DataFrame.from_dict(my_dict)
df.to_csv(path_or_buf='Gestures/train1.csv')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser(description='train speaker specific model')
parser = create_parser(parser)
args = parser.parse_args()

# //read_csv here

# configs = {
#     "audio_to_pose": {"num_keypoints": 136, "processor": "audio_to_pose", "flatten": False, "input_shape": [None, AUDIO_SHAPE]},
#     "audio_to_pose_inference": {"num_keypoints": 136, "processor": "audio_to_pose_inference", "flatten": False, "input_shape": [None, AUDIO_SHAPE]}
# }
cfg = get_config(args.config)
data_loader = a2kData(args,"train", cfg)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = Audio2Keypoint(args).to(device)

# check arguements

generator = Generator().to(device)
discriminator = Discriminator().to(device)
# piv = ImageEncoderPIV().to(device)
PIV = ImageEncoderPIV().to(device)
PV = ImageEncoderPV().to(device)

keypts_regloss = KeyPointsRegLoss(args.reg_loss_type, 'pose')
# Disc_loss = nn.MSELoss(reduction='mean')
Disc_loss = nn.BCELoss()
Enc_loss = nn.TripletMarginLoss(margin=0.5, p=1)
G_Enc_loss = nn.L1Loss()

D_optim = optim.Adam(discriminator.parameters(), lr=args.lr_d)
E_optim = optim.Adam(PIV.parameters(), lr=args.lr_g)
G_optim = optim.Adam(generator.parameters(), lr=args.lr_g)

# TODO : Checkpoint saving, resume training from checkpoint


iteration = 0
start_epoch = 1
epochs = 1

for epoch in range(start_epoch, 2):
    for data in data_loader:
        # print("training start")
        audio, real_pose = [x.to(device) for x in data]
        # print("real_pose")
        # print(real_pose.shape)
        image = real_pose[:, :, 0]
        image = image.unsqueeze(-1)

        # Train with all real examples
        D_training_keypoints = _get_training_keypoints()
        D_real_pose = keypoints_to_train(real_pose, D_training_keypoints)
        D_real_pose = get_sample_output_by_config(D_real_pose, cfg).float()
        real_pose_score = discriminator(D_real_pose)

        input_enc_piv = PIV(image)
        # Train with all fake examples

        fake_pose, input_enc_pv = generator(
            audio, real_pose, image, input_enc_piv)

        D_fake_pose = keypoints_to_train(
            fake_pose, D_training_keypoints).float()

        fake_pose_score = discriminator(D_fake_pose)
        real_pose_score = real_pose_score.squeeze(-1)
        fake_pose_score = fake_pose_score.squeeze(-1)

        print(real_pose_score)
        D_loss_real = Disc_loss(real_pose_score, torch.ones(
            real_pose_score.shape).to(device))
        print("fake")
        D_loss_fake = Disc_loss(fake_pose_score, torch.zeros(
            fake_pose_score.shape).to(device))
        print("D_loss")
        D_loss = D_loss_real + args.lambda_d * D_loss_fake
        print(D_loss)
        D_optim.zero_grad()
        D_loss.backward(retain_graph=True)
        D_optim.step()

        # Generator training

        print("Discriminator")
        fake_pose_score = discriminator(D_fake_pose.detach())
        fake_pose_score = fake_pose_score.squeeze(-1)
        D_loss_fake = Disc_loss(fake_pose_score, torch.ones(
            fake_pose_score.shape).to(device))
        pose_regloss = keypts_regloss(D_real_pose, D_fake_pose)

        # print(fake_pose.shape)
        fake_enc_pv = PV(torch.mean(fake_pose, dim=-1).unsqueeze(-1))

        fake_enc_piv = PIV(torch.mean(fake_pose, dim=-1).unsqueeze(-1))

        real_enc_piv = PIV(torch.mean(fake_pose, dim=-1).unsqueeze(-1))

        enc_loss = Enc_loss(
            input_enc_piv, real_enc_piv.detach(), fake_enc_piv.detach())
        g_enc_loss = G_Enc_loss(input_enc_pv, fake_enc_pv)

        G_loss = args.lambda_gan * D_loss_fake + \
            pose_regloss + args.lambda_enc * g_enc_loss
        G_optim.zero_grad()
        G_loss.backward(retain_graph=True)
        G_optim.step()

        E_optim.zero_grad()
        enc_loss.backward()
        E_optim.step()

        iteration += 1
        print(iteration)

        if iteration % 1 == 0:
            print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                  % (epoch, iteration,
                     G_loss.item(), D_loss.item()))
    if(epoch % 10 == 0):
        torch.save(generator.state_dict(),
                   '/home/btp/pg_btp_1/Aditya/Audio2Keypoint1/tmp/g_w')
        torch.save(PIV.state_dict(),
                   '/home/btp/pg_btp_1/Aditya/Audio2Keypoint1/tmp/e_w')

        # print(f'Iter {iteration} (epoch {epoch}/{epochs}) : Disc loss : {D_loss}, Enc loss : {E_loss}, Gen loss : {G_loss}')
