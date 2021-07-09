import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import ConvLayer, to_motion_delta

class ImageEncoderPIV(nn.Module):
    """
    Pose Invariant Encoder
    """
    def __init__(self):
        super(ImageEncoderPIV, self).__init__()
        self.layers = nn.Sequential(*ConvLayer(136, 128, 3, 1, 1, '2D'),
                                    *ConvLayer(128, 128, 3, 1, 1, '2D'),
                                    *ConvLayer(128, 128, 3, 1, 1, '2D'),
                                    *ConvLayer(128, 128, 3, 1, 1, '2D'),
                                    *ConvLayer(128, 64, 3, 1, 1, '2D'),
                                    *ConvLayer(64, 64, 3, 1, 1, '2D'),
                                    *ConvLayer(64, 64, 3, 1, 1, '2D'),
                                    *ConvLayer(64, 32, 3, 1, 1, '2D'),
                                    *ConvLayer(32, 32, 3, 1, 1, '2D'))

    def forward(self, image):
        """
        Parameters
        ----------
        image       : torch.tensor of shape (B, 136, L)
                      Input image
        Returns
        -------
        img_enc_piv : torch.tensor of shape (B, 32)
                      PIV encoding of the input image
        """
        image = image.unsqueeze(-1) #(B,136,L,1)
        img_enc_piv = self.layers(image).mean(dim = 2).squeeze(-1)
        return img_enc_piv

class ImageEncoderPV(nn.Module):
    """
    Pose Variant Encoder
    """
    def __init__(self):
        super(ImageEncoderPV, self).__init__()
        self.conv1 = ConvLayer(136, 128, 3, 1, 1, '1D', seq = True)
        self.conv2 = ConvLayer(128, 128, 4, 2, 1, '1D', seq = True)
        self.conv3 = ConvLayer(128, 128, 3, 1, 1, '1D', seq = True)

    def forward(self, image):
        """
        Parameters
        ----------
        image : torch.tensor of shape (B, 136, 1)
                Input image

        Returns
        -------
        x     : torch.tensor of shape (B, 128, 2)
                PV encoding of the image
        """
        x = self.conv1(image)
        x = torch.repeat_interleave(x, 2, dim = 2)
        x = self.conv2(x)
        x = torch.repeat_interleave(x, 2, dim = 2)
        x = self.conv3(x)
        return x

class AudioEncoder(nn.Module):

    def __init__(self):
        super(AudioEncoder, self).__init__()
        self.downsampling_blocks1to4 = nn.Sequential(*ConvLayer(1, 64, 3, 1, 1, '2D'),
                                                     *ConvLayer(64, 64, 4, 2, 1, '2D'),
                                                     *ConvLayer(64, 128, 3, 1, 1, '2D'),
                                                     *ConvLayer(128, 64, 4, 2, 1, '2D'),
                                                     *ConvLayer(64, 256, 3, 1, 1, '2D'),
                                                     *ConvLayer(256, 256, 4, 2, 1, '2D'),
                                                     *ConvLayer(256, 256, 3, 1, 1, '2D'),
                                                     *ConvLayer(256, 256, (3, 8), 1, 0, '2D'))
        self.downsampling_blocks5to10 = nn.ModuleList([nn.Sequential(*ConvLayer(256, 256, 3, 1, 1, '1D'),
                                                                     *ConvLayer(256, 256, 3, 1, 1, '1D')),
                                                       ConvLayer(256, 256, 4, 2, 1, '1D', seq = True),
                                                       ConvLayer(256, 256, 4, 2, 1, '1D', seq = True),
                                                       ConvLayer(256, 256, 4, 2, 1, '1D', seq = True),
                                                       ConvLayer(256, 256, 4, 2, 1, '1D', seq = True),
                                                       ConvLayer(256, 256, 4, 2, 1, '1D', seq = True)
                                                      ])
        self.conv = ConvLayer(416, 256, 3, 1, 1, '1D', seq = True)
        self.convs = nn.ModuleList([ConvLayer(256, 256, 3, 1, 1, '1D', seq = True),
                                    ConvLayer(256, 256, 3, 1, 1, '1D', seq = True),
                                    ConvLayer(256, 256, 3, 1, 1, '1D', seq = True),
                                    ConvLayer(256, 256, 3, 1, 1, '1D', seq = True),
                                    ConvLayer(256, 256, 3, 1, 1, '1D', seq = True)])

    def forward(self, audio_spect, img_enc_pv, img_enc_piv, temporal_size):
        """
        Parameters
        ----------
        audio_spect   : torch.tensor of shape (B, 1, 418, 64)
                        Mel spectrogram of audio
        img_enc_pv    : torch.tensor of shape (B, 128, 2)
                        PV encoding
        img_enc_piv   : torch.tensor of shape (B, 32, 2)
                        PIV encoding
        temporal_size : int
                        Size of temporal stack

        Returns
        -------
        x             : torch.tensor of shape (B, 256, 64)
                        Audio encoding
        """
        x = self.downsampling_blocks1to4(audio_spect) #(B,256,50,1)
        x = F.interpolate(x, (temporal_size, 1), mode = 'bilinear', align_corners = False).squeeze(-1) #(B,256,64)
        outs = list()
        for layer in self.downsampling_blocks5to10:
            x = layer(x)
            outs.append(x)
        outs.reverse()
        # (B,256,2), (B,128,2), (B,32,2)
        x = torch.cat([x, img_enc_pv, img_enc_piv], dim = 1) #(B,416,2)
        x = self.conv(x) #(B,256,2)
        for y, layer in zip(outs[1:], self.convs):
            x = torch.repeat_interleave(x, 2, dim = 2)
            x = layer(x + y)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(*ConvLayer(256, 256, 3, 1, 1, '1D'),
                                    *ConvLayer(256, 256, 3, 1, 1, '1D'),
                                    *ConvLayer(256, 256, 3, 1, 1, '1D'),
                                    *ConvLayer(256, 256, 3, 1, 1, '1D'))
        self.logits = nn.Conv1d(256, 136, kernel_size = 1, stride = 1, padding = 0)

    def forward(self, audio_enc):
        """
        Parameters
        ----------
        audio_enc : torch.tensor of shape (B, 256, 64)
                    Audio encoding
        Returns
        -------
        logits    : torch.tensor of shape (B, 136, 64)
                    Fake pose
        """
        dec = self.layers(audio_enc)
        logits = self.logits(dec)
        return logits

class Discriminator(nn.Module):
    def __init__(self, d_input, n_downsampling = 2):
        super(Discriminator, self).__init__()
        self.d_input = d_input
        # d motion or pose
        if d_input == 'motion':
            self.motion_or_pose = lambda x : to_motion_delta(x)
        elif d_input == 'pose':
            self.motion_or_pose = lambda x : x
        elif d_input == 'both':
            self.motion_or_pose = lambda x : torch.cat([x, to_motion_delta(x)], dim = 2)

        if d_input in ['motion', 'both']:
            layers = [nn.ConstantPad1d((1, 2), 0), # TF uses asymmetrical padding
                      *ConvLayer(134, 64, 4, 2, 0, '1D', False)]
        else:
            layers = ConvLayer(134, 64, 4, 2, 1, '1D', False)

        for i in range(1, n_downsampling+1):
            n = min(2**i, 8)
            m = min(2**(i-1), 8)
            if i != n_downsampling:
                layers.extend(ConvLayer(64*m, 64*n, 4, 2, 1, '1D'))
            else:
                layers.extend([nn.ConstantPad1d((1, 2), 0), # TF uses asymmetrical padding
                               *ConvLayer(64*m, 64*n, 4, 1, 0, '1D')])
        layers.extend([nn.ConstantPad1d((1, 2), 0), # TF uses asymmetrical padding
                      nn.Conv1d(64*n, 1, 4, 1, 0)])
        self.layers = nn.Sequential(*layers)

    def forward(self, pose):
        """
        Parameters
        ----------
        pose  : torch.tensor of shape (B, 134, 64)
                Fake or real pose

        Returns
        -------
        score : torch.tensor of shape (B, 16)
                Realism score
        """
        motion_or_pose = self.motion_or_pose(pose) #(B,134,63), (B,134,64) or (B,134,127)
        score = self.layers(motion_or_pose).squeeze(1)
        return score

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.image_encoder_pv = ImageEncoderPV()
        self.audio_encoder = AudioEncoder()
        self.decoder = Decoder()

    def forward(self, audio_spect, img, img_enc_piv, temporal_size):
        """
        Parameters
        ----------
        audio_spect   : torch.tensor of shape (B, 1, 418, 64)
                        Mel spectrogram of audio
        img           : torch.tensor of shape (B, 136, 1)
                        Input image
        img_enc_piv   : torch.tensor of shape (B, 32, 1)
                        PIV encoding of img
        temporal_size : int
                        Size of temporal stack

        Returns
        -------
        fake_pose     : torch.tensor of shape (B, 136, 64)
                        Pose created by generator
        """
        img_enc_pv = self.image_encoder_pv(img) #(B,128,2)
        img_enc_piv = torch.repeat_interleave(img_enc_piv.unsqueeze(2), 2, dim = 2) #(B,32,2)
        audio_enc = self.audio_encoder(audio_spect, img_enc_pv, img_enc_piv, temporal_size) #(B,256,64)
        fake_pose = self.decoder(audio_enc) #(B,136,64)
        return fake_pose
