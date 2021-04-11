import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import ConvLayer, MelSpectrogram,to_motion_delta
from utils import ConvLayer,UpSampling1D
class ImageEncoderPIV(nn.Module):
    """
    Pose Invariant Encoder
    """
    def __init__(self):
        super(ImageEncoderPIV, self).__init__()
        self.layers = nn.Sequential(ConvLayer(136, 128, 3, 1, 1, '2D'),
                                    # ConvLayer(128, 128, 3, 1, 1, '2D'),
                                    ConvLayer(128, 128, 3, 1, 1, '2D'),
                                    ConvLayer(128, 128, 3, 1, 1, '2D'),
                                    ConvLayer(128, 64, 3, 1, 1, '2D'),
                                    ConvLayer(64, 64, 3, 1, 1, '2D'),
                                    ConvLayer(64, 64, 3, 1, 1, '2D'),
                                    ConvLayer(64, 32, 3, 1, 1, '2D'),
                                    ConvLayer(32, 32, 3, 1, 1, '2D'),
                                    ConvLayer(32, 32, 3, 1, 1, '2D'))

    def forward(self, image):
        print("in PIV")
        
        shapes=image.shape
        image=image.view((shapes[0],shapes[1],shapes[2],1))
        image = self.layers(image.float()).mean(dim = 3) # check dim
        image = UpSampling1D(image)
        return image

class ImageEncoderPV(nn.Module):
    """
    Pose Variant Encoder
    """

    def __init__(self):
        super(ImageEncoderPV, self).__init__()
        in_channels=136
        self.conv1 = ConvLayer(in_channels, 128, 3, 1, 1, '1D')
        self.conv2 = ConvLayer(128, 128, 4, 2, 1, '1D') #padding such that same size
        self.conv3 = ConvLayer(128, 128, 3, 1, 1, '1D')

    def forward(self, image):
        image=image.float()
        # image=torch.unsqueeze(image,1)
        # print(image.size())
    
        x=self.conv1(image)
    
        x=UpSampling1D(x)
        # # x = torch.cat([x, x], dim = 2) # check dim
        x = self.conv2(x)
        x=UpSampling1D(x)
        
        # # x = torch.cat([x, x], dim = 2) # check dim
        x = self.conv3(x)
        # print(x.size())
        return x

class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()
        in_channels=1
        padding=1
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
                                                       ConvLayer(in_channels, 256, 4, 2, padding, '1D') #padding such that same size
                                                      ])
        self.downsampling_blocks5_downsampling_true = ConvLayer(256, 256, 3, 1, 1, '1D')
        self.downsampling_blocks5_downsampling_false =ConvLayer(256, 256, 4, 2, 1, '1D')

        self.conv = ConvLayer(in_channels, 256, 3, 1, 1, '1D')
        self.convs = nn.ModuleList([ConvLayer(in_channels, 256, 3, 1, 1, '1D'),
                                    ConvLayer(256, 256, 3, 1, 1, '1D'),
                                    ConvLayer(256, 256, 3, 1, 1, '1D'),
                                    ConvLayer(256, 256, 3, 1, 1, '1D'),
                                    ConvLayer(256, 256, 3, 1, 1, '1D')])

    def forward(self, audio, pose, img_enc_pv, img_enc_piv):
        x = self.downsampling_blocks1to4(audio)
        x = F.interpolate(x, (pose.shape[1], 1), mode = 'bilinear', align_corners = False)#.squeeze(2) # check dim
        x=x.squeeze(-1)
        print(";ks;")
        print(x.shape)
        fifth_block=self.downsampling_blocks5_downsampling_false(x)
        print(fifth_block.shape)
        fifth_block=self.downsampling_blocks5_downsampling_false(fifth_block)
        
        sixth_block=self.downsampling_blocks5_downsampling_true(fifth_block)
        
        seventh_block=self.downsampling_blocks5_downsampling_true(sixth_block)
        
        eight_block=self.downsampling_blocks5_downsampling_true(seventh_block)
        
        ninth_block=self.downsampling_blocks5_downsampling_true(eight_block)
        
        tenth_block=self.downsampling_blocks5_downsampling_true(ninth_block)

        print("conact_shapes")
        print(tenth_block.shape)
        print(img_enc_pv.shape)
        print(img_enc_piv.shape)
        tenth_block=torch.cat([tenth_block,img_enc_pv,img_enc_piv],dim=1)
        tenth_block=self.downsampling_blocks5_downsampling_false(tenth_block)
       
        ninth_block=self.UpSampling1D(tenth_block)+ninth_block
        ninth_block=self.downsampling_blocks5_downsampling_false(ninth_block)
       
        eight_block=self.UpSampling1D(ninth_block)+eight_block
        eight_block=self.downsampling_blocks5_downsampling_false(eight_block)

        seventh_block = self.UpSampling1D(eight_block) + seventh_block
        seventh_block = self.downsampling_blocks5_downsampling_false(seventh_block)

        sixth_block = self.UpSampling1D(seventh_block) + sixth_block
        sixth_block = self.downsampling_blocks5_downsampling_false(sixth_block)

        fifth_block = self.UpSampling1D(sixth_block) + fifth_block
        audio_encoding=self.downsampling_blocks5_downsampling_false(fifth_block)
        return audio_encoding

class AudioEncoder2(nn.Module):

    def __init__(self):
        super(AudioEncoder2, self).__init__()
        in_channels=1
        padding=1
        self.downsampling_blocks1to4 = nn.Sequential(ConvLayer(in_channels, 64, 3, 1, 1, '2D'),PrintSize(),
                                                     ConvLayer(64, 64, 4, 2,(210,33),'2D'),PrintSize(), #padding such that same size
                                                     ConvLayer(64, 128, 3, 1, 1, '2D'),PrintSize(),
                                                     ConvLayer(128, 64, 4, 2, (210,33), '2D'), #padding such that same size
                                                     ConvLayer(64, 256, 3, 1, 1, '2D'),PrintSize(),
                                                     ConvLayer(256, 256, 4, 2, (210,33), '2D'), #padding such that same size
                                                     ConvLayer(256, 256, 3, 1, 1, '2D'),PrintSize(),
                                                     ConvLayer(256, 256, (3, 8), 1, (1,4), '2D'),PrintSize()) #padding such that same size
        self.downsampling_blocks5to10 = nn.ModuleList([nn.Sequential(ConvLayer(in_channels, 256, 3, 1, 1, '1D'),
                                                                     ConvLayer(256, 256, 3, 1, 1, '1D')),
                                                       ConvLayer(in_channels, 256, 4, 2, padding, '1D'), #padding such that same size
                                                       ConvLayer(in_channels, 256, 4, 2, padding, '1D'), #padding such that same size,
                                                       ConvLayer(in_channels, 256, 4, 2, padding, '1D'), #padding such that same size
                                                       ConvLayer(in_channels, 256, 4, 2, padding, '1D'), #padding such that same size
                                                       ConvLayer(in_channels, 256, 4, 2, padding, '1D') #padding such that same size
                                                      ])
        self.downsampling_blocks10_downsampling_false = ConvLayer(416, 256, 3, 1, 1, '1D')
        self.downsampling_blocks5_downsampling_true = ConvLayer(256, 256, 4, 2, 1, '1D')
        self.downsampling_blocks5_downsampling_false =ConvLayer(256, 256, 3, 1, 1, '1D')

        self.conv = ConvLayer(in_channels, 256, 3, 1, 1, '1D')
        self.convs = nn.ModuleList([ConvLayer(in_channels, 256, 3, 1, 1, '1D'),
                                    ConvLayer(256, 256, 3, 1, 1, '1D'),
                                    ConvLayer(256, 256, 3, 1, 1, '1D'),
                                    ConvLayer(256, 256, 3, 1, 1, '1D'),
                                    ConvLayer(256, 256, 3, 1, 1, '1D')])

    def forward(self, audio, pose, img_enc_pv, img_enc_piv):
        print("In forward of AudioEncoder")
        x = self.downsampling_blocks1to4(audio)
        print("downsample_done")
        print(x.shape)
        x = F.interpolate(x, (pose.shape[2], 1), mode = 'bilinear', align_corners = False)#.squeeze(2) # check dim
        x=x.squeeze(-1)
        print("check downsampling")
        print(x.shape)
        fifth_block=self.downsampling_blocks5_downsampling_false(x)
        print(fifth_block.shape)
        fifth_block=self.downsampling_blocks5_downsampling_false(fifth_block)
        
        sixth_block=self.downsampling_blocks5_downsampling_true(fifth_block)
        
        seventh_block=self.downsampling_blocks5_downsampling_true(sixth_block)
        
        eight_block=self.downsampling_blocks5_downsampling_true(seventh_block)
        
        ninth_block=self.downsampling_blocks5_downsampling_true(eight_block)
        
        tenth_block=self.downsampling_blocks5_downsampling_true(ninth_block)

        # print("conact_shapes")
        print("10")
        print(tenth_block.shape)
        print(img_enc_pv.shape)
        print(img_enc_piv.shape)
        tenth_block=torch.cat([tenth_block,img_enc_pv,img_enc_piv],dim=1)
        tenth_block=self.downsampling_blocks10_downsampling_false(tenth_block)
        print(tenth_block.shape)
        
        print("check downsampling again")
        print(ninth_block.shape)
        
        ninth_block=UpSampling1D(tenth_block)+ninth_block
        print(ninth_block.shape)
        ninth_block=self.downsampling_blocks5_downsampling_false(ninth_block)
        print(ninth_block.size())
       
        eight_block=UpSampling1D(ninth_block)+eight_block
        eight_block=self.downsampling_blocks5_downsampling_false(eight_block)

        seventh_block = UpSampling1D(eight_block) + seventh_block
        seventh_block = self.downsampling_blocks5_downsampling_false(seventh_block)

        sixth_block = UpSampling1D(seventh_block) + sixth_block
        sixth_block = self.downsampling_blocks5_downsampling_false(sixth_block)

        fifth_block = UpSampling1D(sixth_block) + fifth_block
        decoder_input=self.downsampling_blocks5_downsampling_false(fifth_block)
        print("decoder_input")
        print(decoder_input.shape)
        return decoder_input
       
        
        # outs = list()
        # for layer in self.downsampling_blocks5to10:
        #     x = layer(x)
        #     outs.append(x)
        # outs.reverse()
        # x = torch.cat([x, img_enc_pv, img_enc_piv], dim = 2) #check dim
        # x = self.conv(x)
        # for :
        #     x
        # for y, layer in zip(outs[1:], self.convs):
        #     x = CatAndAdd(x, y, layer)
        # return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        input_channels=256
        self.layers = nn.Sequential(ConvLayer(input_channels, 256, 3, 1, 1, '1D'),PrintSize(),
                                    ConvLayer(256, 256, 3, 1, 1, '1D'),PrintSize(),
                                    ConvLayer(256, 256, 3, 1, 1, '1D'),PrintSize(),
                                    ConvLayer(256, 256, 3, 1, 1, '1D'),PrintSize())
        self.logits = nn.Conv1d(256, 136, kernel_size = 1, stride = 1, padding = 0)

    def forward(self, audio_enc):
        dec = self.layers(audio_enc)
        logits = self.logits(dec)
        PrintSize()
        return logits

class PrintSize(nn.Module):
  def __init__(self):
    super(PrintSize, self).__init__()
    
  def forward(self, x):
    print(x.shape)
    return x

class Discriminator(nn.Module):
    def __init__(self, d_input='pose'):
        super(Discriminator, self).__init__()
        self.d_input = d_input
        # d motion or pose
        if d_input == 'motion':
            self.motion_or_pose = lambda x : to_motion_delta(sel)
        elif d_input == 'pose':
            self.motion_or_pose = lambda x : x
        elif d_input == 'both':
            self.motion_or_pose = lambda x : torch.cat([x, to_motion_delta(x)], dim = 1) # check dim
        n_downsampling=2
        in_channels=136
        m=0
        n=0
        layers = list()
        # layers.append(PrintSize())
        layers.append(ConvLayer(in_channels, 64, 4, 2, 33 ,'1D', False))
        # layers.append(PrintSize())


        for i in range(1 ,n_downsampling+1):
            nf_mult = min(2**(i), 8)
            m = min(2**(i-1), 8)
            if(i!=n_downsampling):
                # layers.append(PrintSize())
                layers.append(ConvLayer(64*m, 64*nf_mult, 4, 2 ,34, '1D'))
                layers.append(PrintSize())
                 #padding such that same size
        layers.append(ConvLayer(64*m, 64*nf_mult, 4,1,1,'1D'))
        
        layers.append(PrintSize())
        layers.append(nn.Conv1d((64*(2**i)), 1, 2, 1, padding=1)) #padding such that same size
        layers.append(PrintSize())
        layers.append(nn.Sigmoid())
        # layers.append(nn.Conv1d(64*nf_mult, 1, 4, 1, padding=0))
        self.layers = nn.Sequential(*layers)

    def forward(self, pose):
        motion_or_pose = self.motion_or_pose(pose)
        # motion_or_pose=motion_or_pose
        return self.layers(motion_or_pose)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.image_encoder_pv = ImageEncoderPV()
        # self.image_encoder_piv=ImageEncoderPIV()
        self.audio_encoder = AudioEncoder2()
        self.decoder = Decoder()

    def forward(self, audio, pose, image,image_enc_piv):
        image_enc_pv =self.image_encoder_pv(image)
        # image_enc_piv = torch.cat([image_enc_piv, image_enc_piv], dim = 1) # check dim
        # audio_input2=tf_mel_spectograms(audio)
        audio_input = MelSpectrogram(audio)
        
        audio_enc = self.audio_encoder(audio_input, pose, image_enc_pv,image_enc_piv)
        out = self.decoder(audio_enc)
        return out,image_enc_pv

class Generator2(nn.Module):
    def __init__(self):
        super(Generator2, self).__init__()
        # self.image_encoder_pv = ImageEncoderPV()
        # self.image_encoder_piv=ImageEncoderPIV()
        self.audio_encoder = AudioEncoder2()
        self.decoder = Decoder()

    def forward(self, audio, pose,image):
        # image_enc_pv =self.image_encoder_pv(image)
        # image_enc_piv=self.image_encoder_piv(image)
        # image_enc_piv=UpSampling1D(image_enc_piv)
        
        # image_enc_piv = torch.cat([image_enc_piv, image_enc_piv], dim = 1) # check dim

        # audio_input2=tf_mel_spectograms(audio)
        audio_input = MelSpectrogram(audio)
        audio_enc = self.audio_encoder(audio_input, pose, image, image)
        print("print decoder shapes")
        out = self.decoder(audio_enc)
        return out



# check_tensor=torch.rand((1,64,136))

# checkPIV=ImageEncoderPIV()
# checkPV=ImageEncoderPV()
# decode=Decoder()
# # discriminator=Discriminator()
# # print(discriminator(check_tensor).size())

# out=checkPIV(check_tensor)
# out1=checkPV(check_tensor)
# print(decode(check_tensor).size())
# print(out.size())
# print(out1.size())
