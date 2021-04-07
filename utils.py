import torch
import functools
# from tensorflow.contrib.signal.python.ops import window_ops
import tensorflow as tf
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as F_au
from common.consts import CHIN_KEYPOINTS, LEFT_BROW_KEYPOINTS, RIGHT_BROW_KEYPOINTS, NOSE_KEYPOINTS,\
    LEFT_EYE_KEYPOINTS, RIGHT_EYE_KEYPOINTS, OUTER_LIP_KEYPOINTS, INNER_LIP_KEYPOINTS, POSE_SAMPLE_SHAPE, G_SCOPE, D_SCOPE, E_SCOPE, SR

def _get_training_keypoints():
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

def ConvLayer(in_channels, out_channels, kernel_size, stride, padding, type, norm = True):
    assert type in ['1D', '2D']
    if type == '1D':
        return nn.Sequential(nn.Conv1d(in_channels = in_channels,
                              out_channels = out_channels,
                              kernel_size = kernel_size,
                              stride = stride,
                              padding = padding),
                         nn.BatchNorm1d(out_channels),
                         nn.LeakyReLU(0.2))
    else:
        return nn.Sequential(nn.Conv2d(in_channels = in_channels,
                              out_channels = out_channels,
                              kernel_size = kernel_size,
                              stride = stride,
                              padding = padding),
                         nn.BatchNorm2d(out_channels),
                         nn.LeakyReLU(0.2))



def CatAndAdd(x, y, layer):
    return layer(torch.cat([x, x], dim = 1) + y) # check dim

def MelSpectrogram(audio):
    print(audio.shape)
    stft = torch.stft(audio, n_fft = 512, hop_length = 160, win_length = 400,
                      window = torch.hann_window(window_length=400,periodic = True),
                      center = False).abs()
    stft=stft[:,:,:,0]
    mel_spect_input = F_au.create_fb_matrix(stft.shape[1], n_mels = 64,
                                            f_min = 125.0, f_max = 7500.0,
                                            sample_rate = 16000)
    print(stft.shape)
    print(mel_spect_input.shape)
    input_data = torch.tensordot(stft, mel_spect_input, dims = [[1],[0]])
    print(input_data.shape)
    input_data = torch.log(input_data + 1e-6).unsqueeze(1)
    print(input_data.shape)
    return input_data
def tf_mel_spectograms(audio):
    stft = tf.signal.stft(
        audio,
        400,
        160,
        fft_length=512,
        window_fn=tf.signal.hann_window,
        pad_end=False,
        name=None
    )
    stft = tf.abs(stft)
    print("stft")
    print(stft.shape)
    mel_spect_input = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=64,
        num_spectrogram_bins=tf.shape(stft)[2],
        sample_rate=16000,
        lower_edge_hertz=125.0,
        upper_edge_hertz=7500.0,
        dtype=tf.float32,
        name=None
    )
    print("mel_spect_input")
    print(mel_spect_input.shape)
    input_data = tf.tensordot(stft, mel_spect_input, 1)
    print("input_data")
    print(input_data.shape)
    input_data = tf.log(input_data + 1e-6)
    
    input_data = tf.expand_dims(input_data, -1)

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
def UpSampling1D(input):
    input=torch.repeat_interleave(input,2,2)
    return input
class KeyPointsRegLoss(nn.Module):
    def __init__(self, type, loss_on, lambda_motion):
        super(KeyPointsRegLoss, self).__init__()
        assert type in ['l1', 'l2']
        self.loss = nn.L1Loss() if type == 'L1' else nn.MSELoss()
        assert loss_on in ['pose', 'motion', 'both']
        self.loss_on = loss_on
        # self.lambda = lambda_motion

    def forward(self, real_keypts, fake_keypts):
        loss = 0
        # Do we really need to flatten before loss computation???
        if self.loss_on in ['pose', 'both']:
            loss = self.loss(real_keypts.view(-1), fake_keypts.view(-1))
        if self.loss_on in ['motion', 'both']:
            real_keypts_motion = to_motion_delta(real_keypts).view(-1)
            fake_keypts_motion = to_motion_delta(fake_keypts).view(-1)
            loss += self.loss(real_keypts_motion, fake_keypts_motion) #* self.lambda
        return loss
