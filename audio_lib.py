import datetime
import os
import subprocess
import resampy
import numpy as np
import pandas as pd
import librosa
from scipy.io import wavfile

from consts import SECOND, MINUTE, HOUR

def frame_number_to_seconds(num, fps):
    return float(num + 1) / fps

def time_in_seconds_to_datetime(time_seconds):
    return str(datetime.timedelta(seconds=time_seconds))

def get_string_to_timedata(time):
    return datetime.datetime.strptime(time, '%H:%M:%S.%f')

def get_timedata_to_seconds(td):
    td = get_string_to_timedata(td) if isinstance(td, str) else td
    return td.hour * HOUR + td.minute * MINUTE + td.second * SECOND + float(td.microsecond) / 1000000.

def get_pose_path(base_path, speaker, pose_fn):
    return os.path.join(base_path, speaker, 'keypoints_simple', pose_fn)

def get_frame_path(base_path, speaker, frame_fn):
    return os.path.join(base_path, speaker, 'frames', frame_fn)

def get_video_path(base_path, speaker, video_fn):
    return os.path.join(base_path, speaker, 'videos', video_fn)

def save_audio_sample_from_video(vid_path, audio_out_path, audio_start, audio_end, sr=44100):
    if not (os.path.exists(os.path.dirname(audio_out_path))):
        os.makedirs(os.path.dirname(audio_out_path))
    cmd = 'ffmpeg -i "%s" -ss %s -to %s -ab 160k -ac 2 -ar %s -vn "%s" -y -loglevel warning' % (
    vid_path, audio_start, audio_end, sr, audio_out_path)
    subprocess.call(cmd, shell=True)

def save_audio_sample(audio, audio_out_path, input_sr=16000, output_sr=44100):
    if not (os.path.exists(os.path.dirname(audio_out_path))):
        os.makedirs(os.path.dirname(audio_out_path))
    audio = resampy.resample(audio, input_sr, output_sr)
    wavfile.write(audio_out_path, output_sr, audio)

def get_audio_from_video_by_time(interval_start, interval_end, speaker_name,
                                 video_fn, base_dataset_path, audio_out_path, sr):
    save_audio_sample_from_video(get_video_path(base_dataset_path, speaker_name, video_fn), audio_out_path,
                                 interval_start, interval_end)
    wav, sr = librosa.load(audio_out_path, sr, mono = True)
    return wav

def get_img_pos_wav(sample, video_fn, audio_out_path, base_dataset_path, sr):
    interval_start, interval_end = get_interval_start_end(sample)
    speaker_name = sample.iloc[0]['speaker']

    wav = get_audio_from_video_by_time(interval_start, interval_end, speaker_name, video_fn, base_dataset_path,
                                       audio_out_path, sr)
    poses = np.array([np.loadtxt(get_pose_path(base_dataset_path, row['speaker'], row['pose_fn'])) for _, row in
                      sample.iterrows()])
    img_fns = np.array(
        [get_frame_path(base_dataset_path, row['speaker'], row['frame_fn']) for _, row in sample.iterrows()])
    return img_fns, poses, wav

def get_interval_start_end(sample):
    interval_start = str(pd.to_datetime(sample.iloc[0]['pose_dt']).time())
    interval_end = str(pd.to_datetime(sample.iloc[-1]['pose_dt']).time())
    return interval_start, interval_end
