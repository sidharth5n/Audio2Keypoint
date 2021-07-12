from __future__ import print_function, division
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

from utils import preprocess_to_relative, normalize_relative_keypoints

class VoxKP(data.Dataset):
    def __init__(self, args, split):
        df = pd.read_csv(args.train_csv)
        if args.speaker != None:
            df = df[df['speaker'] == args.speaker]
        self.df = df[df['dataset'] == split]
        self.flatten = args.flatten
        self.indices = [*range(len(self.df))]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        audio_spect : torch.tensor of shape (1, 418, 64)
        real_pose : torch.tensor of shape (136, 64)
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

# # import pandas as pd
# # df = pd.read_csv("train.csv")
# AUDIO_SHAPE = 67267
# configs = {
#     "input_shape": [None, AUDIO_SHAPE],
#     "audio_to_pose": {"num_keypoints": 136, "processor": "audio_to_pose", "flatten": False, "input_shape": [None, AUDIO_SHAPE]},
#     "audio_to_pose_inference": {"num_keypoints": 136, "processor": "audio_to_pose_inference", "flatten": False, "input_shape": [None, AUDIO_SHAPE]}
# }
#
# # path = r"C:\Users\sidhu\Desktop\train.csv"
# mydata = VoxKP(args, "train", configs["audio_to_pose"]).__getitem__([0])
