import torch
import torch.nn as nn
from torch import optim
from model import Audio2Keypoint
# from dataset import VoxKP
from utils import KeyPointsRegLoss, keypoints_to_train,_get_training_keypoints
from config import get_config
from common.pose_logic_lib import translate_keypoints, get_sample_output_by_config
AUDIO_SHAPE = 67267
import pandas as pd
my_dict={"audio_fn":["./Gestures/human/train/audio/id0004462OEFEevKvs00001-00:00:00.040000-00:00:05.360000.wav"],
"dataset":["train"],"end":["0:00:02.560000"],"interval_id":["id037619diQL48epnM00019"],
"pose_fn":["./Gestures/human/train/npz/id0004.npz"],
"speaker":["human"],"start":["0:00:00.040000"],"video_fn":["id03761/9diQL48epnM/00019.mp4"]}
df=pd.DataFrame.from_dict(my_dict)
df.to_csv(path_or_buf='Gestures/train1.csv')
from dataload import a2kData
import argparse

from config import create_parser

parser = argparse.ArgumentParser(description='train speaker specific model')
parser = create_parser(parser)
args = parser.parse_args()

# //read_csv here

configs = {
    "audio_to_pose": {"num_keypoints": 136, "processor": "audio_to_pose", "flatten": False, "input_shape": [None, AUDIO_SHAPE]},
    "audio_to_pose_inference": {"num_keypoints": 136, "processor": "audio_to_pose_inference", "flatten": False, "input_shape": [None, AUDIO_SHAPE]}
}
data_loader =  a2kData(df,"train",configs["audio_to_pose"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Audio2Keypoint(args).to(device)


keypts_regloss = KeyPointsRegLoss(args.reg_loss_type, args.d_input, args.lambda_motion_reg_loss)
Disc_loss = nn.MSELoss()
Enc_loss = nn.TripletMarginLoss(margin = 0.5, p = 1)
G_Enc_loss = nn.L1Loss()

D_optim = optim.Adam(model.discriminator.parameters(), lr = args.lr_d)
E_optim = optim.Adam(model.encoder.parameters(), lr = args.lr_g)
G_optim = optim.Adam(model.generator.parameters(), lr = args.lr_g)

# TODO : Checkpoint saving, resume training from checkpoint

iteration = 0
start_epoch=1
epochs=1
cfg = get_config(args.config)
for epoch in range(start_epoch, 2):
    for data in data_loader:
        audio, real_pose = [x.to(device) for x in data]
        image=real_pose[:,0]
        image=torch.unsqueeze(image,1)
        img_enc_piv, fake_pose, real_enc, fake_enc, real_pose_score, fake_pose_score = model(image.float(), audio, real_pose)
        
        # remove base keypoint which is always [0,0]. Keeping it may ruin GANs training due discrete problems. etc.
        training_keypoints = _get_training_keypoints()

        train_real_pose = keypoints_to_train(real_pose, training_keypoints)
        train_real_pose = get_sample_output_by_config(train_real_pose, cfg)
        train_fake_pose = keypoints_to_train(fake_pose, training_keypoints)

        # Regression loss on motion or pose
        pose_regloss = keypts_regloss(train_real_pose, train_fake_pose)

        # Compute discriminator loss and update D
        D_loss_real = Disc_loss(real_pose_score.new_ones(), real_pose_score)
        D_loss_fake = Disc_loss(fake_pose_score.new_zeros(), fake_pose_score)
        D_loss = D_loss_real + args.lambda_d * D_loss_fake
        D_optim.zero_grad()
        D_loss.backward()
        D_optim.step()

        # Loss for training the generator from the global D - have I fooled the global D?
        D_loss_fake = Disc_loss(fake_pose_score.new_ones(), fake_pose_score)
        # encoder loss and train encoder
        E_enc_loss = Enc_loss(img_enc_piv, real_enc, fake_enc)
        E_loss = pose_regloss + D_loss_fake + E_enc_loss
        E_optim.zero_grad()
        E_loss.backward()
        E_optim.step()

        # loss for generatar encoding
        G_enc_loss = G_Enc_loss(img_enc_piv, fake_enc)
        # sum up ALL the losses for training the generator
        G_loss = pose_regloss + (args.lambda_gan * D_loss_fake) + (args.lambda_enc * G_enc_loss)
        G_optim.zero_grad()
        G_loss.backward()
        G_optim.step()

        iteration += 1
        print(f'Iter {iteration} (epoch {epoch}/{epochs}) : Disc loss : {D_loss}, Enc loss : {E_loss}, Gen loss : {G_loss}')
