import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as F_au
from common.consts import CHIN_KEYPOINTS, LEFT_BROW_KEYPOINTS, RIGHT_BROW_KEYPOINTS, NOSE_KEYPOINTS,\
    LEFT_EYE_KEYPOINTS, RIGHT_EYE_KEYPOINTS, OUTER_LIP_KEYPOINTS, INNER_LIP_KEYPOINTS

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

def MelSpectrogram(audio):
    """
    Computes log mel spectrogram of audio input.

    Parameters
    ----------
    audio      : torch.tensor of shape (B, config.input_shape[1])

    Returns
    -------
    input_data : torch.tensor of shape (B, 1, 418, 64)
    """
    device = audio.device
    stft = torch.stft(audio, n_fft = 512, hop_length = 160, win_length = 400,
                      window = torch.hann_window(window_length = 400, periodic = True).to(device),
                      center = False, return_complex = True).abs()
    mel_spect_input = F_au.create_fb_matrix(stft.shape[1], n_mels = 64,
                                            f_min = 125.0, f_max = 7500.0,
                                            sample_rate = 16000).to(device)
    input_data = torch.tensordot(stft, mel_spect_input, dims = [[1], [0]])
    input_data = torch.log(input_data + 1e-6).unsqueeze(1)
    return input_data

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
    def __init__(self, type, loss_on, alpha_motion):
        super(KeyPointsRegLoss, self).__init__()
        assert type in ['L1', 'L2']
        self.loss = nn.L1Loss() if type == 'L1' else nn.MSELoss()
        assert loss_on in ['pose', 'motion', 'both']
        self.loss_on = loss_on
        self.alpha = alpha_motion

    def forward(self, real_keypts, fake_keypts):
        loss = 0
        # Do we really need to flatten before loss computation???
        if self.loss_on in ['pose', 'both']:
            loss = self.loss(real_keypts.view(-1), fake_keypts.view(-1))
        if self.loss_on in ['motion', 'both']:
            real_keypts_motion = to_motion_delta(real_keypts).view(-1)
            fake_keypts_motion = to_motion_delta(fake_keypts).view(-1)
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
