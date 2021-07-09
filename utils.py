import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchaudio.functional as F_au

from consts import CHIN_KEYPOINTS, LEFT_BROW_KEYPOINTS, RIGHT_BROW_KEYPOINTS, NOSE_KEYPOINTS,\
    LEFT_EYE_KEYPOINTS, RIGHT_EYE_KEYPOINTS, OUTER_LIP_KEYPOINTS, INNER_LIP_KEYPOINTS, mean, std, scale_factor

try:
    import cPickle as pickle
except:
    import pickle

def save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def ConvLayer(in_channels, out_channels, kernel_size, stride, padding, conv_type, norm = True, seq = False):
    assert conv_type in ['1D', '2D']
    conv = nn.Conv1d if conv_type == '1D' else nn.Conv2d
    norm_ = nn.BatchNorm1d if conv_type == '1D' else nn.BatchNorm2d
    layers = [conv(in_channels = in_channels,
                              out_channels = out_channels,
                              kernel_size = kernel_size,
                              stride = stride,
                              padding = padding),
              norm_(out_channels) if norm else nn.Identity(),
              nn.LeakyReLU(0.2)]
    return nn.Sequential(*layers) if seq else layers

# def MelSpectrogram(audio):
#     """
#     Computes log mel spectrogram of audio input.
#
#     Parameters
#     ----------
#     audio      : torch.tensor of shape (B, config.input_shape[1])
#
#     Returns
#     -------
#     input_data : torch.tensor of shape (B, 1, 418, 64)
#     """
#     device = audio.device
#     stft = torch.stft(audio, n_fft = 512, hop_length = 160, win_length = 400,
#                       window = torch.hann_window(window_length = 400, periodic = True).to(device),
#                       center = False, return_complex = True).abs()
#     mel_spect_input = F_au.create_fb_matrix(stft.shape[1], n_mels = 64,
#                                             f_min = 125.0, f_max = 7500.0,
#                                             sample_rate = 16000).to(device)
#     input_data = torch.tensordot(stft, mel_spect_input, dims = [[1], [0]])
#     input_data = torch.log(input_data + 1e-6).unsqueeze(1)
#     return input_data

def keypoints_to_train(poses, arr):
    shape = poses.shape
    poses = poses.permute(0, 2, 1)
    reshaped = poses.reshape((shape[0], shape[2], 2, 68))
    # required_keypoints = torch.gather(reshaped, index = arr, dim = 3)
    required_keypoints = reshaped[...,arr]
    required_keypoints = required_keypoints.reshape((shape[0], shape[2], 2*len(arr)))
    return required_keypoints.permute(0, 2, 1)

def to_motion_delta(pose_batch):
    shape = pose_batch.shape
    reshaped = pose_batch.view((-1, 2, shape[1] // 2, 64))
    diff = reshaped[..., 1:] - reshaped[..., :-1]
    diff = diff.view((-1, shape[1], 63))
    return diff

class KeyPointsRegLoss(nn.Module):
    def __init__(self, type, loss_on, alpha_motion, reduction = 'mean'):
        super(KeyPointsRegLoss, self).__init__()
        self.loss = nn.L1Loss(reduction = reduction) if type == 1 else nn.MSELoss(reduction = reduction)
        self.loss_on = loss_on
        self.alpha = alpha_motion

    def forward(self, real_pose, fake_pose, keypoints, train_ratio):
        nbatches = real_pose.shape[0]
        loss = 0
        real_pose = keypoints_to_train(real_pose, keypoints)
        real_keypts = get_sample_output_by_train_ratio(real_pose, train_ratio)
        fake_keypts = keypoints_to_train(fake_pose, keypoints)
        # Do we really need to flatten before loss computation???
        if self.loss_on in ['pose', 'both']:
            loss = self.loss(real_keypts.view(nbatches, -1), fake_keypts.view(nbatches, -1))
        if self.loss_on in ['motion', 'both']:
            real_keypts_motion = to_motion_delta(real_keypts).reshape(nbatches, -1)
            fake_keypts_motion = to_motion_delta(fake_keypts).reshape(nbatches, -1)
            loss += self.loss(real_keypts_motion, fake_keypts_motion) * self.alpha
        return loss

def get_training_keypoints():
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
    training_keypoints = torch.LongTensor(training_keypoints)
    return training_keypoints

def get_sample_output_by_train_ratio(x, train_ratio):
    if train_ratio is not None:
        num_timesteps = x.shape[1]
        return x[:, conditioned_timesteps(train_ratio, num_timesteps):]
    return x

def conditioned_timesteps(train_ratio, num_timesteps):
    return int(train_ratio * num_timesteps)

def preprocess_to_relative(k, reshape = True, num_keypoints = 68):
    """
    Converts to relative.

    Parameters
    ----------
    k             : np.ndarray of shape (64, 2, 68)
                    Key points
    reshape       : bool, optional
                    Whether to reshape. Default is True.
    num_keypoints : int, optional
                    No. of keypoints. Default is 68.
    Returns
    -------
    relative      : np.ndarray of shape (64, 136) if reshape is True else (64, 2, 68)
                    Relative keypoints
    """
    reshaped = k.reshape((-1, 2, num_keypoints))
    relative = reshaped - reshaped[:, :, 27:28]
    if reshape:
        return relative.reshape((-1, num_keypoints * 2))
    return relative

def normalize_relative_keypoints(k, speaker):
    """
    Mean variance normalizes the key points.

    Parameters
    ----------
    k          : np.ndarray of shape (64, 136)
    speaker    : str

    Returns
    -------
    normalized : np.ndarray of shape (64, 136)
    """
    normalized = (k - mean) / (std + np.finfo(float).eps)
    return normalized

def translate_keypoints(keypoints, shift):
    """
    keypoints : torch.tensor of shape (64, 2, 68)
    """
    return keypoints + shift.view(1, 2, 1)

def de_normalize_relative_keypoints(k, scale_to_jon):
    """
    Parameters
    ----------
    k : torch.tensor of shape (B, 136, 64)
        Key points

    Returns
    -------
    k : torch.tensor of shape (B, 136, 64)
        De-normalized key points
    """
    from consts import mean, std, scale_factor
    std = torch.from_numpy(std).view(1, -1, 1).to(k)
    mean = torch.from_numpy(mean).view(1, -1, 1).to(k)
    keypoints = (k * std + torch.finfo(float).eps) + mean
    if scale_to_jon:
        keypoints = scale_factor * keypoints
    return keypoints


def decode_pose_normalized_keypoints(encoded_keypoints, shift, scale_to_jon=True):
    """
    Parameters
    ----------
    encoded_keypoints : torch.tensor of shape (B, 136, 64)

    Returns
    -------
    denormalized : torch.tensor of shape (B, 64, 2, 68)
    """
    batch_size = encoded_keypoints.shape[0]
    denormalized = de_normalize_relative_keypoints(encoded_keypoints, scale_to_jon)
    denormalized = denormalized.view(batch_size, 2, 68, -1).permute(0, 3, 1, 2)
    return translate_keypoints(denormalized, shift)


def decode_pose_normalized_keypoints_no_scaling(encoded_keypoints, shift):
    return decode_pose_normalized_keypoints(encoded_keypoints, shift, scale_to_jon=False)

def compute_pck(pred, gt, alpha=0.02):
    """
    Computes percentage of correct keypoints.

    Parameters
    ----------
    pred  : torch.tensor of shape (B, 64, 2, 67)
            Predicted keypoints on NxMxK where N is number of samples, M is of
            shape 2, corresponding to X,Y and K is the number of keypoints to be
            evaluated on
    gt    :  similarly
            Ground truth key points
    alpha : float, optional
            Parameters controlling the scale of the region around the image
            multiplied by the max(H,W) of the person in the image. We follow
            https://www.cs.cmu.edu/~deva/papers/pose_pami.pdf and set it to 0.1

    Return
    ------
    pck_score : torch.tensor of shape ()
                Mean prediction score
    """
    pck_radius = compute_pck_radius(gt, alpha)
    keypoint_overlap = (torch.norm(gt-pred, p = 2, dim = 2) <= (pck_radius))
    pck_score = keypoint_overlap.sum() / keypoint_overlap.numel()
    return pck_score

def compute_pck_radius(gt, alpha):
    """
    Parameters
    ----------
    gt                    : torch.tensor of shape (B, 64, 2, 67)
                            Ground truth key points

    Returns
    -------
    max_axis_per_keypoint : torch.tensor of shape (B, 64, 67)
    """
    width = (gt[:, :, 0:1].max(3)[0] - gt[:, :, 0:1].min(3)[0]).abs()
    height = (gt[:, :, 1:2].max(3)[0] - gt[:, :, 1:2].min(3)[0]).abs()
    max_axis = torch.cat([width, height], axis = 2).max(2)[0]
    max_axis_per_keypoint = torch.tile(max_axis.unsqueeze(-1), [1, 67])
    return max_axis_per_keypoint * alpha
