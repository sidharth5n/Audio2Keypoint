import numpy as np
import pandas as pd
import librosa
import scipy
import os
import argparse

def preprocess_x(x, audio_length):
    """
    Zero pads audio file at the end to fixed size.

    Parameters
    ----------
    x : numpy.ndarray of shape (L,)

    Returns
    -------
    x : numpy.ndarray of shape (audio_length,)
    """
    if len(x) > audio_length:
        x = x[:audio_length]
    elif len(x) < audio_length:
        x = np.pad(x, [0, audio_length - len(x)], mode='constant', constant_values=0)
    return x

def mel_spectrogram(audio, args):
    """
    Computes log mel spectrogram of audio input.

    Parameters
    ----------
    audio     : np.ndarray of shape (audio_length, )
    args      :

    Returns
    -------
    mel_spect : np.ndarray of shape (418, 64) for audio_length 67267
                Mel Spectrogram of audio input in log scale
    """
    stft = librosa.stft(audio, n_fft = args.n_fft, hop_length = args.hop_length,
                        win_length = args.win_length,
                        window = scipy.signal.windows.hann(M = args.win_length, sym = False),
                        center = False)
    stft = np.abs(stft)
    stft_to_mel = librosa.filters.mel(sr = 16000, n_fft = args.n_fft,
                                      n_mels = args.n_mels, fmin = 125.0,
                                      fmax = 7500.0, htk = True, norm = None)
    mel_spect = np.tensordot(stft, stft_to_mel, axes = [[0], [1]])
    mel_spect = np.log(mel_spect + 1e-6)
    return mel_spect

def process_audio_to_spectogram(args):

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    df = pd.read_csv(args.csv)

    for i in range(len(df)):
        row = df.iloc[i]
        arr = np.load(row['pose_fn'])
        audio = arr['audio']
        audio = preprocess_x(audio, args.audio_length)
        mel_spect = mel_spectrogram(audio, args)
        # file_name = os.path.split(row['pose_fn'])[1][:-4]
        np.savez(row['pose_fn'], imgs = arr['imgs'], pose = arr['pose'], audio = arr['audio'], melspect = mel_spect)
        # np.save(os.path.join(args.output_dir, file_name + '.npy'), mel_spect)

        if i % 1000 == 0:
            print(f'Processed {i}/{len(df)} files.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_length', type = int, default = 67267,
                    help = 'Size of audio')
    parser.add_argument('--n_mels', type = int, default = 64,
                    help = 'No. of Mel bands to generate')
    parser.add_argument('--n_fft', type = int, default = 512,
                    help = 'Length of windowed signal after padding with zeros')
    parser.add_argument('--win_length', type = int, default = 400,
                    help = 'Length of window before padding with zeros to match n_fft')
    parser.add_argument('--hop_length', type = int, default = 160,
                    help = 'No. of audio samples between adjacent STFT columns')

    parser.add_argument('--csv', type = str, default = 'Gestures/train.csv')
    parser.add_argument('--output_dir', type = str, default = 'Gestures/human/train/spectrogram',
                    help = 'Directory where extracted spectrograms are to be saved')

    args = parser.parse_args()
    process_audio_to_spectogram(args)
