import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as F_au
from common.consts import CHIN_KEYPOINTS, LEFT_BROW_KEYPOINTS, RIGHT_BROW_KEYPOINTS, NOSE_KEYPOINTS,\
    LEFT_EYE_KEYPOINTS, RIGHT_EYE_KEYPOINTS, OUTER_LIP_KEYPOINTS, INNER_LIP_KEYPOINTS

def ConvLayer(in_channels, out_channels, kernel_size, stride, padding, type, norm = True, seq = False):
    assert type in ['1D', '2D']
    conv = nn.Conv1d if type == '1D' else nn.Conv2d
    norm_ = nn.BatchNorm1d if type == '1D' else nn.BatchNorm2d
    layers = [conv(in_channels = in_channels,
                              out_channels = out_channels,
                              kernel_size = kernel_size,
                              stride = stride,
                              padding = padding),
              norm_(out_channels) if norm else lambda x: x,
              nn.LeakyReLU(0.2)]
    return nn.Sequential(*layers) if seq else layers

def CatAndAdd(x, y, layer):
    return layer(torch.repeat_interleave(x, 2, dim = 2) + y)

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
    stft = torch.stft(audio, n_fft = 512, hop_length = 160, win_length = 400,
                      window = torch.hann_window(window_length = 400, periodic = True),
                      center = False, return_complex = True).abs().transpose(2,1)
    mel_spect_input = F_au.create_fb_matrix(stft.shape[2], n_mels = 64,
                                            f_min = 125.0, f_max = 7500.0,
                                            sample_rate = 16000)
    input_data = torch.tensordot(stft, mel_spect_input, dims = 1)
    input_data = torch.log(input_data + 1e-6).unsqueeze(1)
    return input_data

def keypoints_to_train(poses, arr):
    shape = poses.shape
    reshaped = poses.view((shape[0], shape[1], 2, 68))
    required_keypoints = torch.gather(reshaped, index = arr, dim = 3)
    required_keypoints = required_keypoints.view((shape[0], shape[1], 2*len(arr)))
    return required_keypoints

def to_motion_delta(pose_batch):
    shape = pose_batch.shape
    reshaped = pose_batch.view((-1, 64, 2, shape[-1]/2))
    diff = reshaped[:, 1:] - reshaped[:, :-1]
    diff = diff.view((-1, 63, shape[-1]))
    return diff

class KeyPointsRegLoss(nn.Module):
    def __init__(self, type, loss_on, lambda_motion):
        super(KeyPointsRegLoss, self).__init__()
        assert type in ['L1', 'L2']
        self.loss = nn.L1Loss() if type == 'L1' else nn.MSELoss()
        assert loss_on in ['pose', 'motion', 'both']
        self.loss_on = loss_on
        self.lambda = lambda_motion

    def forward(self, real_keypts, fake_keypts):
        loss = 0
        # Do we really need to flatten before loss computation???
        if self.loss_on in ['pose', 'both']:
            loss = self.loss(real_keypts.view(-1), fake_keypts.view(-1))
        if self.loss_on in ['motion', 'both']:
            real_keypts_motion = to_motion_delta(real_keypts).view(-1)
            fake_keypts_motion = to_motion_delta(fake_keypts).view(-1)
            loss += self.loss(real_keypts_motion, fake_keypts_motion) * self.lambda
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
