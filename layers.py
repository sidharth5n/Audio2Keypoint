import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import ConvLayer, CatAndAdd, MelSpectrogram

class ImageEncoderPIV(nn.Module):
    """
    Pose Invariant Encoder
    """
    def __init__(self):
        super(ImageEncoderPIV, self).__init__()
        self.layers = nn.Sequential(ConvLayer(input_channels, 128, 3, 1, 1, '2D'),
                                    ConvLayer(128, 128, 3, 1, 1, '2D'),
                                    ConvLayer(128, 128, 3, 1, 1, '2D'),
                                    ConvLayer(128, 128, 3, 1, 1, '2D'),
                                    ConvLayer(128, 64, 3, 1, 1, '2D'),
                                    ConvLayer(64, 64, 3, 1, 1, '2D'),
                                    ConvLayer(64, 64, 3, 1, 1, '2D'),
                                    ConvLayer(64, 32, 3, 1, 1, '2D'),
                                    ConvLayer(32, 32, 3, 1, 1, '2D'))

    def forward(self, image):
        return self.layers(image).mean(dim = 2)

class ImageEncoderPV(nn.Module):
    """
    Pose Variant Encoder
    """
    def __init__(self):
        super(ImageEncoderPV, self).__init_()
        self.conv1 = ConvLayer(in_channels, 128, 3, 1, 1, '1D')
        self.conv2 = ConvLayer(in_channels, 128, 4, 2, padding = , '1D') #padding such that same size
        self.conv3 = ConvLayer(in_channels, 128, 3, 1, 1, '1D')

    def forward(self, image):
        x = self.conv1(image)
        x = torch.cat([x, x], dim = 1)
        x = self.conv2(x)
        x = torch.cat([x, x], dim = 1)
        x = self.conv3(x)
        return x

class AudioEncoder(nn.Module):

    def __init__(self):
        super(AudioEncoder, self).__init__()
        self.downsampling_blocks1to4 = nn.Sequential(ConvLayer(in_channels, 64, 3, 1, 1, '2D'),
                                                     ConvLayer(64, 64, 4, 2, padding, '2D'), #padding such that same size
                                                     ConvLayer(64, 128, 3, 1, 1, '2D'),
                                                     ConvLayer(128, 64, 4, 2, padding, '2D'), #padding such that same size
                                                     ConvLayer(64, 256, 3, 1, 1, '2D'),
                                                     ConvLayer(256, 256, 4, 2, padding, '2D'), #padding such that same size
                                                     ConvLayer(256, 256, 3, 1, 1, '2D'),
                                                     ConvLayer(256, 256, (3, 8), 1, padding, '2D')) #padding such that same size
        self.downsampling_blocks5to10 = nn.ModuleList([nn.Sequential(ConvLayer(in_channels, 256, 3, 1, 1, '1D'),
                                                                     ConvLayer(256, 256, 3, 1, 1, '1D')),
                                                       ConvLayer(in_channels, 256, 4, 2, padding, '1D'), #padding such that same size
                                                       ConvLayer(in_channels, 256, 4, 2, padding, '1D'), #padding such that same size,
                                                       ConvLayer(in_channels, 256, 4, 2, padding, '1D'), #padding such that same size
                                                       ConvLayer(in_channels, 256, 4, 2, padding, '1D'), #padding such that same size
                                                       ConvLayer(in_channels, 256, 4, 2, padding, '1D')) #padding such that same size
                                                      ])
        self.conv = ConvLayer(in_channels, 256, 3, 1, 1, '1D')
        self.convs = nn.ModuleList([ConvLayer(in_channels, 256, 3, 1, 1, '1D'),
                                    ConvLayer(256, 256, 3, 1, 1, '1D'),
                                    ConvLayer(256, 256, 3, 1, 1, '1D'),
                                    ConvLayer(256, 256, 3, 1, 1, '1D'),
                                    ConvLayer(256, 256, 3, 1, 1, '1D')])

    def forward(self, audio, pose, img_enc_pv, img_enc_piv):
        x = self.downsampling_blocks1to4(audio)
        x = F.upsample(x, (pose.shape[1], 1), mode = 'bilinear').squeeze(2)
        outs = list()
        for layer in self.downsampling_blocks5to10:
            x = layer(x)
            outs.append(x)
        outs.reverse()
        x = torch.cat([x, img_enc_pv, img_enc_piv], dim = 2)
        x = self.conv(x)
        for y, layer in zip(outs[1:], self.convs):
            x = CatAndAdd(x, y, layer)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(ConvLayer(input_channels, 256, 3, 1, 1, '1D'),
                                    ConvLayer(input_channels, 256, 3, 1, 1, '1D'),
                                    ConvLayer(input_channels, 256, 3, 1, 1, '1D'),
                                    ConvLayer(input_channels, 256, 3, 1, 1, '1D'))
        self.logits = nn.Conv1d(input_channels, 136, kernel_size = 1, stride = 1, padding = 0)

    def forward(self, audio_enc):
        dec = self.layers(audio_enc)
        logits = self.logits(dec)
        return logits

class Discriminator(nn.Module):
    def __init__(self):
        super(D_patchgan, self).__init__()
        layers = list()
        for i in range(n_downsampling+1):
            if i == 0:
                layers.append(ConvLayer(in_channels, 64, 4, 2, padding, '1D', False)) #padding such that same size
            else:
                n = min(2**i, 8)
                m = min(2**(i-1), 8)
                layers.append(ConvLayer(64*m, 64*n, 4, 1 if i == n_downsampling else 2, padding, '1D')) #padding such that same size
        layers.append(nn.Conv1d(64*n, 1, 4, 1, padding)) #padding such that same size
        self.layers = nn.Sequential(layers)

    def forward(self, x_pose):
        return self.layers(x_pose)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.image_encoder_pv = ImageEncoderPV()
        self.audio_encoder = AudioEncoder()
        self.decoder = Decoder()

    def forward(self, audio, pose, image, image_enc_piv):
        image_enc_pv = self.image_encoder_pv(image)
        image_enc_piv = torch.cat([image_enc_piv, image_enc_piv], dim = 1)
        audio_input = MelSpectrogram(audio)
        audio_enc = self.audio_encoder(audio_input, pose, img_enc_pv, img_enc_piv)
        out = self.decoder(audio_enc)
        return out
