from __future__ import print_function, division
import numpy as np
import librosa
import pandas as pd
import torch
import torch.utils.data as data

from utils import preprocess_to_relative, normalize_relative_keypoints, pad_audio, mel_spectrogram
from consts import SR, FPS, AUDIO_SHAPE

class AudioSample(data.Dataset):
    def __init__(self, args):
        self.flatten = args.flatten

    def preprocess_audio_and_image(self, args, device):
        audio, _ = librosa.load(args.audio_path, sr = SR, mono = True)
        pose_shape = int(FPS * float(audio.shape[0]) / SR)
        padded_pose_shape = pose_shape + (2**6) - pose_shape % (2**6)
        padded_audio_shape = int(padded_pose_shape * SR / FPS)
        padded_audio = np.pad(audio, [0, padded_audio_shape - audio.shape[0]], mode = 'reflect')
        total_div = padded_audio_shape // 40960
        for i in range(total_div):
            pad_audio_use = padded_audio[i*40960: (i+1)*40960]
            mel_spect = mel_spectrogram(pad_audio(pad_audio_use, AUDIO_SHAPE), args)
            np.save(os.path.join('data', 'temp', f'{i}.npy'), mel_spect)

        import face_alignment
        from skimage import io
        
        input = io.imread(args.image_path)
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device = device)
        pred = fa.get_landmarks(input)[0]
        pred = pred.T
        np.save(os.path.join('data', 'temp', 'driving.npy'), pred)

        return pred, pose_shape, padded_pose_shape

    def __len__(self):
        return len(os.listdir(os.path.join('data', 'temp')))

    def __getitem__(self, idx):
        audio_spect = np.load(os.path.join('data', 'temp', f'{i}.npy'))
        audio_spect = torch.from_numpy(audio_spect)
        if idx == 0:
            driving_input = np.load(os.path.join('data', 'temp', 'driving.npy'))
            driving_input = preprocess_to_relative(driving_input)
            driving_input = normalize_relative_keypoints(driving_input)
            driving_input = driving_input.flatten() if self.flatten else np.swapaxes(driving_input, 1, 0)
            driving_input = torch.from_numpy(driving_input.astype(np.float32))
        else:
            driving_input = None
        return audio_spect, driving_input

class VoxKP(data.Dataset):
    def __init__(self, args, split):
        df = pd.read_csv(args.train_csv)
        self.df = df[df['dataset'] == split]
        self.flatten = args.flatten
        self.indices = [*range(len(self.df))]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        audio_spect : torch.tensor of shape (1, 418, 64)
        real_pose   : torch.tensor of shape (136, 64)
        """
        row = self.df.iloc[self.indices[idx]]
        arr = np.load(row['pose_fn'])
        audio_spect = arr['melspect']
        audio_spect = np.expand_dims(audio_spect, 0) # Adding a channel (1, 418, 64)
        real_pose = preprocess_to_relative(arr['pose'])
        real_pose = normalize_relative_keypoints(real_pose)
        real_pose = real_pose.flatten() if self.flatten else np.swapaxes(real_pose, 1, 0)
        real_pose = torch.from_numpy(real_pose.astype(np.float32))
        audio_spect = torch.from_numpy(audio_spect)
        return audio_spect, real_pose

class SubsetSampler(data.sampler.Sampler):

    def __init__(self, end, start = 0):
        """
        Parameters
        ----------
        end   : int
                Last index
        start : int, optional
                Starting index. Default is 0.
        """
        self.start = start
        self.end = end

    def __iter__(self):
        start = self.start
        self.start = 0
        return (i for i in range(start, self.end))

    def __len__(self):
        return self.end - self.start

class DataLoader:

    def __init__(self, args, split, params = None, length = 0, num_workers = 0):
        self.split = split
        self.shuffle = True if split == 'train' else False
        self.dataset = VoxKP(args, split)
        self.iterator = 0

        if params is not None:
            self.load_state_dict(params)

        num_samples = length if length > 0 else len(self)

        sampler = SubsetSampler(num_samples, self.iterator)

        self.loader = data.DataLoader(dataset = self.dataset, batch_size = args.batch_size,
                                      sampler = sampler, num_workers = num_workers)

    def __iter__(self):
        for batch_data in self.loader:
            self.iterator += batch_data[0].shape[0]
            if self.iterator >= len(self):
                self.iterator = 0
                if self.shuffle:
                    random.shuffle(self.loader.dataset.indices)
            yield batch_data

    def __len__(self):
        return len(self.dataset)

    def load_state_dict(self, params):
        if 'split' in params:
            assert self.split == params['split']
        self.dataset.indices = params.get('indices', self.dataset.indices)
        self.iterator = params.get('iterator', self.iterator)

    def state_dict(self):
        return {'indices' : self.loader.dataset.indices, 'iterator' : self.iterator, 'split' : self.split}

    def get_df(self):
        return self.dataset.df
