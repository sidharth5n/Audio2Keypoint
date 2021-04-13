

from common.pose_logic_lib import translate_keypoints

import numpy as np
# from common.audio_repr import raw_repr
from common.consts import SR
from config import create_parser, get_config
# from common.pose_plot_lib import save_video_from_audio_video, save_side_by_side_video
import os
import argparse
from layers import Generator,ImageEncoderPIV
from dataload import a2kData
import pandas as pd
import torch
AUDIO_SHAPE = 67267

df=pd.read_csv("Gestures/train1.csv")
configs = {
    "audio_to_pose": {"num_keypoints": 136, "processor": "audio_to_pose", "flatten": False, "input_shape": [None, AUDIO_SHAPE]},
    "audio_to_pose_inference": {"num_keypoints": 136, "processor": "audio_to_pose_inference", "flatten": False, "input_shape": [None, AUDIO_SHAPE]}
}
data_loader =  a2kData(df,"train",configs["audio_to_pose"])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for data in data_loader:
    print("inference started")
    audio, real_pose = [x.to(device) for x in data]
    
# def run_inference(args):
#     if not(os.path.exists(args.output_path)):
#         os.makedirs(args.output_path)
    # audio_fn = args.audio
    # audio, _ = raw_repr(audio_fn, SR)
    # pose_shape = int(25 * float(audio.shape[0]) / SR)
    # padded_pose_shape = pose_shape + (2**6) - pose_shape%(2**)
    # padded_audio_shape = padded_pose_shape * SR / 25
    # padded_audio = np.pad(audio, [0, padded_audio_shape - audio.shape[0]], mode='reflect')
    # cfg = get_config(args.config)

    model = Generator().to(device)
    model.load_state_dict(torch.load('/home/btp/pg_btp_1/Aditya/Audio2Keypoint1/tmp/g_w'))
    encoder = ImageEncoderPIV().to(device)
    encoder.load_state_dict(torch.load('/home/btp/pg_btp_1/Aditya/Audio2Keypoint1/tmp/e_w'))
    image=real_pose[:,:,0]
    image=image.unsqueeze(-1)
    input_enc_piv=encoder(image)
    output_keypoints = model(audio,real_pose,image,input_enc_piv)

    # real_pose = real_pose[:,:,1]
    # real_pose = real_pose.reshape((1,2,68))
    
    # print(real_pose.shape)
    # if args.checkpoint:
    #     pgan.restore(args.checkpoint, scope_list=["generator", "encoder", "discriminator"])
    # else:

    #     print "No Checkpoint provided."
    # padded_pred_kpts = pgan.predict_audio(padded_audio, cfg, args.speaker, [0,0])
    # padded_pred_kpts = translate_keypoints(padded_pred_kpts, [900, 290])

    # input = io.imread(args.input_image)
    # fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda:0')
    # pred = fa.get_landmarks(input)[0]
    # pred = pred.T
    # pred = pred.reshape((1, 136))
    # input_kp = pred

    # final_pred_kpts = np.empty([padded_pose_shape, 2, 68])

    # total_div = padded_audio_shape//40960
    # for i in range(total_div):
    #     pad_audio_use = padded_audio[i*40960: (i+1)*40960]
    #     print("predict_audio")
    #     print(pad_audio_use.shape)
    #     print(padded_audio.shape)
    #     #print('------------------------------clear')
    #     padded_pred_kpts = pgan.predict_audio(pad_audio_use, input_kp, cfg, args.speaker, [0,0])
    #     final_pred_kpts[i*64:(i+1)*64] = padded_pred_kpts
    #     print padded_pred_kpts.shape()
    #     input_kp = padded_pred_kpts[-1].reshape((1, 136))
    #     print(padded_pred_kpts.shape)
    # pred_kpts = final_pred_kpts[:pose_shape]
    np.save('check1.npy',real_pose.cpu().detach().numpy())
    # print('SAVED!')
    # tmp_output_dir = 'tmp/'
    # mute = os.path.join(tmp_output_dir, 'mute_pred.mp4')
    # output_fn = os.path.join(args.output_path, 'output.mp4')
    # save_side_by_side_video(tmp_output_dir, pred_kpts, pred_kpts, mute, delete_tmp=False)
    # save_video_from_audio_video(audio_fn, mute, output_fn)