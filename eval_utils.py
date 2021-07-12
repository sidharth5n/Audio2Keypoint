import os
import datetime
import numpy as np
import torch
import subprocess

from utils import decode_pose_normalized_keypoints, decode_pose_normalized_keypoints_no_scaling, compute_pck, translate_keypoints
from pose_plot_lib import save_side_by_side_video, save_video_from_audio_video
from audio_lib import save_audio_sample

def eval_split(model, loader, loss_fns, args, device, scaling = True, shift_pred=(900, 290), shift_gt=(1900, 280)):
    decode_pose = decode_pose_normalized_keypoints if scaling else decode_pose_normalized_keypoints_no_scaling
    fake_keypoints_list = []
    gt_keypoints_list = []
    losses = []

    shift_pred = torch.tensor(shift_pred).to(device)
    shift_gt = torch.tensor(shift_gt).to(device)

    for data in loader:
        audio_spect, real_pose = [x.to(device) for x in data]
        fake_pose, fake_pose_score, fake_enc, img_enc_piv = model(audio_spect, real_pose, mode = 'predict')
        G_loss = compute_gen_loss(args, loss_fns, real_pose, fake_pose, fake_pose_score, fake_enc, img_enc_piv, model.keypoints)
        fake_keypoints = decode_pose(fake_pose, shift_pred)
        gt_keypoints = decode_pose(real_pose, shift_gt)
        fake_keypoints_list.append(fake_keypoints)
        gt_keypoints_list.append(gt_keypoints)
        losses.append(G_loss)

    fake_keypoints_list = torch.cat(fake_keypoints_list, dim = 0)
    gt_keypoints_list = torch.cat(gt_keypoints_list, dim = 0)
    losses = torch.cat(losses, dim = 0)
    pck_loss = compute_pck(fake_keypoints_list[..., 1:], gt_keypoints_list[..., 1:])
    avg_loss = losses.mean().item()

    fake_keypoints_list = translate_keypoints(fake_keypoints_list, fake_keypoints_list.new_tensor([150, 150]))
    fake_keypoints_list = fake_keypoints_list.detach().cpu().numpy()
    gt_keypoints_list = translate_keypoints(gt_keypoints_list, gt_keypoints_list.new_tensor([450, 150]))
    gt_keypoints_list = gt_keypoints_list.detach().cpu().numpy()

    if args.dump_videos:
        tc = datetime.datetime.now()
        base_path = os.path.join(args.dump_path, args.id, '%s' %
                                 (str(tc).replace('.', '-').replace(' ', '--').replace(':', '-')))
        os.makedirs(base_path)
        save_prediction_video_by_percentiles(loader.get_df(), fake_keypoints_list,
                                             gt_keypoints_list,
                                             base_path,
                                             train_ratio=args.train_ratio,
                                             limit=32, loss = losses.detach().cpu().numpy())

    return pck_loss, avg_loss

def compute_gen_loss(args, losses, real_pose, fake_pose, fake_pose_score, fake_enc, img_enc_piv, keypoints):
    """
    Parameters
    ----------
    losses          : dict
                      Discriminator, Generator Encoding and Regression Loss
    real_pose       : torch.tensor of shape (B, 136, 64)
                      Ground truth pose
    fake_pose       : torch.tensor of shape (B, 136, 64)
                      Pose created by generator
    fake_pose_score : torch.tensor of shape (B, 16)
                      Realism score assigned by discriminator
    fake_enc        : torch.tensor of shape (B, 32)
                      PIV encoding of fake pose
    img_enc_piv     : torch.tensor of shape (B, 32)
                      PIV encoding of input image

    Returns
    -------
    G_loss          : torch.tensor of shape (B, )
    """
    D_loss_fake = losses['D'](torch.ones_like(fake_pose_score), fake_pose_score).sum(-1)
    pose_regloss = losses['K'](real_pose, fake_pose, keypoints, args.train_ratio).sum(-1)
    G_enc_loss = losses['G'](img_enc_piv, fake_enc).sum(-1)
    G_loss = pose_regloss + (args.lambda_gan * D_loss_fake) + (args.lambda_enc * G_enc_loss)
    return G_loss



def save_prediction_video_by_percentiles(df, keypoints_pred, keypoints_gt, save_path, loss_percentile_bgt=95,
                                         loss_percentile_smt=5, train_ratio=None, limit=None, loss=None):
    """
    keypoints_pred : numpy.ndarray of shape (B, 64, 68, 2)
    keypoints_gt   : numpy.ndarray of shape (B, 64, 68, 2)
    """
    if limit is None:
        limit = len(df)

    if loss_percentile_bgt != None:
        thres = np.percentile(loss, loss_percentile_bgt)
        indices = np.where(loss > thres)[0]
        save_prediction_video(df.iloc[indices], keypoints_pred[indices], keypoints_gt[indices],
                              os.path.join(save_path, str(loss_percentile_bgt)),
                              loss=loss[indices], limit = limit // 2)

    if loss_percentile_smt != None:
        thres = np.percentile(loss, loss_percentile_smt)
        indices = np.where(loss < thres)[0]
        save_prediction_video(df.iloc[indices], keypoints_pred[indices], keypoints_gt[indices],
                              os.path.join(save_path, str(loss_percentile_smt)),
                              loss=loss[indices], limit = limit // 2)

    save_prediction_video(df, keypoints_pred, keypoints_gt, os.path.join(save_path, 'random'),
                          loss = loss, limit = limit // 2)

def save_prediction_video(df, keypoints_pred, keypoints_gt, save_path, limit=None, loss=None):
    """
    keypoints_pred : torch.tensor of shape (B, 64, 2, 68)
    keypoints_gt   : torch.tensor of shape (B, 64, 2, 68)
    """
    if limit == None:
        limit = len(df)

    for i in range(min(len(df), limit)):
        row = df.iloc[i]
        keypoints1 = keypoints_pred[i]
        keypoints2 = keypoints_gt[i]

        dir_name = os.path.join(save_path, str(row['interval_id']))

        if not (os.path.exists(dir_name)):
            os.makedirs(dir_name)

        video_fn = os.path.basename(row['video_fn']).split('.')[0]
        interval_id = row['interval_id']
        temp_otpt_fn = os.path.join(dir_name, f'{interval_id}.mp4')
        otpt_fn = os.path.join(save_path, f'{video_fn}_{interval_id}_{row["start"]}_{row["end"]}_{{loss}}.mp4')

        save_side_by_side_video(dir_name, keypoints1, keypoints2, temp_otpt_fn, delete_tmp=False)
        audio = np.load(row['pose_fn'])['audio']
        audio_out_path = os.path.join('tmp', 'audio_cache', f'{row["interval_id"]}_{row["start"]}_{row["end"]}.wav')
        save_audio_sample(audio, audio_out_path, 16000, 16000)
        if loss is not None:
            otpt_fn = otpt_fn.format(loss=loss[i])
        save_video_from_audio_video(audio_out_path, temp_otpt_fn, otpt_fn)
        subprocess.call('rm -R "%s"' % (dir_name), shell=True)
