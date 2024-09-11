import logging
import os
import librosa
import math
import sys
import warnings
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

def default_collate_fn(data):
    vec_seq, audio, video_id, start_idx, end_idx, std, mean, source_images = zip(*data)

    vec_seq = default_collate(vec_seq)
    audio = default_collate(audio)

    video_id = default_collate(video_id)
    start_idx = default_collate(start_idx)
    end_idx = default_collate(end_idx)
    std = default_collate(std)
    mean = default_collate(mean)
    source_images = default_collate(source_images)

    return vec_seq, audio, video_id, start_idx, end_idx, std, mean, source_images


class SpeechMotionDataset(Dataset):
    def __init__(self, data_dir, n_poses, subdivision_stride, pose_resampling_fps, audio_sampling_rate, normalize=False):
        super(SpeechMotionDataset).__init__()
        self.sr = audio_sampling_rate
        self.data_dir = data_dir
        self.n_poses = n_poses
        self.subdivision_stride = subdivision_stride
        self.pose_resampling_fps = pose_resampling_fps

        self.audio_sample_length = int(self.n_poses / self.pose_resampling_fps * self.sr)

        self.sample_keypoints_list = []
        self.sample_audio_list = []
        self.video_id_list = []
        self.start_idx_list = []
        self.end_idx_list = []
        self.source_image_paths = []
        self.std = torch.load('std.pt')
        self.mean = torch.load('mean.pt')

        keys = []
        logging.info("Reading data '{}'...".format(data_dir))
        keypoints_dir = os.path.join(data_dir, 'keypoints')
        audio_dir = os.path.join(data_dir, 'audio')
        for file in os.listdir(keypoints_dir):
            vid_id, ext = os.path.splitext(file)
            keypoint_file = os.path.join(keypoints_dir, file)# + ext)
            audio_file = os.path.join(audio_dir, vid_id + '.mp3')
            audio, sr = librosa.load(audio_file, sr=self.sr)
            keypoints = torch.load(keypoint_file, map_location=torch.device('cpu'))
            keys.append(keypoints)

            num_subdivisions = math.floor((len(keypoints) - self.n_poses) / subdivision_stride) + 1

            for i in range(num_subdivisions):
                start_idx = i * subdivision_stride
                end_idx = start_idx + self.n_poses

                sample_keypoints = keypoints[start_idx:end_idx]

                if normalize:
                    sample_keypoints = (sample_keypoints - self.mean) / self.std

                audio_start = math.floor(start_idx / len(keypoints) * len(audio))
                audio_end = audio_start + self.audio_sample_length

                if audio_end > len(audio):
                    n_padding = audio_end - len(audio)
                    padded_data = np.pad(audio, (0, n_padding), mode='symmetric')
                    sample_audio = padded_data[audio_start:audio_end]
                else:
                    sample_audio = audio[audio_start:audio_end]

                source_image_path = f'{data_dir}/images/{vid_id}/{start_idx:0>4}.png'
                self.source_image_paths.append(source_image_path)

                self.sample_audio_list.append(sample_audio)
                self.sample_keypoints_list.append(sample_keypoints.view(-1, 50*2))
                self.video_id_list.append(vid_id)
                self.start_idx_list.append(start_idx)
                self.end_idx_list.append(end_idx)

    def __len__(self):
        return len(self.sample_audio_list)

    def __getitem__(self, index):
        return self.sample_keypoints_list[index], self.sample_audio_list[index], \
            self.video_id_list[index], self.start_idx_list[index], self.end_idx_list[index], \
            self.std, self.mean, self.source_image_paths[index]