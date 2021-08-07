from __future__ import print_function, division

import os
import numpy as np
import pandas as pd
import librosa
import argparse

from consts import SR, FPS, AUDIO_SHAPE
from audio_lib import get_timedata_to_seconds
from utils import pad_audio, mel_spectrogram

def get_timedata_to_image_frame(time):
    return int(get_timedata_to_seconds(time) * FPS)

def get_timedata_to_audio_frame(time):
    bias = '00:00:00.040000'
    return = int((get_timedata_to_seconds(time) - get_timedata_to_seconds(bias)) * SR)

def form_temporal_stack(args):

    if not os.path.exists(os.path.join(args.data_dir, 'npz')):
        os.mkdir(os.path.join(args.data_dir, 'npz'))

    df = pd.read_csv(args.csv)
    audio_path = None

    for i in range(len(df)):
        row = df.iloc[i]

        start = get_timedata_to_image_frame(row['start']) + row['delta_image_start']
        end = get_timedata_to_image_frame(row['end']) + row['delta_image_end'] + 1
        frames = []
        for j in range(start, end):
            file = os.path.join(args.data_dir, 'keypoints', row['video_fn'].replace('.mp4', ''), f'{j}.txt')
            frames.append(np.loadtxt(file, delimiter = ' '))
        frames = np.stack(frames, axis = 0)
        assert frames.shape[0] == 64, f"Temporal length is {frames.shape[0]} (should be 64) for index {i}"

        start = get_timedata_to_audio_frame(row['start']) + row['delta_audio_start']
        end = get_timedata_to_audio_frame(row['end']) + row['delta_audio_end'] + 1
        temp_path = os.path.join(args.data_dir, row['audio_fn'])
        if audio_path is None or temp_path != audio_path:
            audio_path = temp_path
            audio, _ = librosa.load(audio_path, SR, mono = True)
        audio_padded = pad_audio(audio[start:end], args.audio_length)
        audio_spect = mel_spectrogram(audio_padded, args)

        file = os.path.join(args.data_dir, row['pose_fn'])
        np.savez_compressed(file, pose = frames, melspect = audio_spect)

        if (i % 1000) == 0:
            print(f"Processed {i}/{len(df)}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_length', type = int, default = AUDIO_SHAPE,
                    help = 'Size of audio')
    parser.add_argument('--n_mels', type = int, default = 64,
                    help = 'No. of Mel bands to generate')
    parser.add_argument('--n_fft', type = int, default = 512,
                    help = 'Length of windowed signal after padding with zeros')
    parser.add_argument('--win_length', type = int, default = 400,
                    help = 'Length of window before padding with zeros to match n_fft')
    parser.add_argument('--hop_length', type = int, default = 160,
                    help = 'No. of audio samples between adjacent STFT columns')

    parser.add_argument('--csv', type = str, default = 'data/infos.csv',
                    help = "Path to the dataset csv file")
    parser.add_argument('--data_dir', type = str, default = 'data',
                    help = "Root directory where data is placed")

    args = parser.parse_args()
    form_temporal_stack(args)
