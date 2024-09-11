import shutil

from torch.utils.data import DataLoader
import datetime
import librosa
import lmdb
import logging
import math
import numpy as np
import os
import pickle
import pprint
import pyarrow
import random
import soundfile as sf
import sys
import time
import torch
import torch.nn.functional as F
import yaml

from data_loader.data_preprocessor import DataPreprocessor
from data_loader.tpsm_data_loader import SpeechMotionDataset, default_collate_fn
from model.embedding_space_evaluator import EmbeddingSpaceEvaluator
from model.evaluator import Evaluator
from utils.average_meter import AverageMeter
from utils.train_utils import set_logger, set_random_seed
from parse_args_diffusion import parse_args

from model.motion_ae import MotionAE

from skimage import io, img_as_float32, img_as_ubyte
from skimage.draw import disk
import imageio
import matplotlib.pyplot as plt
from textwrap import wrap
import matplotlib.animation as animation
import subprocess

from tps.modules.inpainting_network import InpaintingNetwork
from tps.modules.keypoint_detector import KPDetector
# from tps.modules.bg_motion_predictor import BGMotionPredictor
from tps.modules.dense_motion import DenseMotionNetwork
from tps.logger import Visualizer

from model.pose_diffusion import PoseDiffusion

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sigma = 0.1
thres = 0.01


def load_checkpoint_and_model(checkpoint_path, _device='cpu'):
    print('loading checkpoint {}'.format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=_device)
    args = checkpoint['args']
    epoch = checkpoint['epoch']
    lang_model = checkpoint['lang_model']
    speaker_model = checkpoint['speaker_model']
    pose_dim = checkpoint['pose_dim']
    print('epoch {}'.format(epoch))

    if args.model == 'pose_diffusion' or args.model == 'edge':
        print("init diffusion model")
        model = PoseDiffusion(args).to(_device)

    model.load_state_dict(checkpoint['state_dict'])

    return args, model, lang_model, speaker_model, pose_dim

from scipy import linalg
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """ from https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py """
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def evaluate(test_data_loader, model, evaluator, args, pose_dim, kp_detector, inpainting_network, dense_motion_network):
    if evaluator:
        evaluator.reset()
        # losses = AverageMeter('loss')

    # init embed net
    ckpt = torch.load('results/ae_checkpoint_499.bin', map_location=device)
    latent_dim = ckpt['latent_dim']
    net = MotionAE(pose_dim, latent_dim).to(device)
    net.load_state_dict(ckpt['state_dict'])
    net.train(False)

    context_feat_list = []
    real_feat_list = []
    generated_feat_list = []
    recon_err_diff = []
    cos_err_diff = []

    bcs = 0
    bc_i = 0

    joint_mae = AverageMeter('mae_on_joint')
    accel = AverageMeter('accel')
    bc = AverageMeter('bc')
    start = time.time()

    with torch.no_grad():
        for iter_idx, data in enumerate(test_data_loader, 0):
            print("testing {}/{}".format(iter_idx, len(test_data_loader)))
            target_vec, in_audio, video_id, start_idx, end_idx, std, mean, source_image_paths = data
            batch_size = target_vec.size(0)

            in_audio = in_audio.to(device)
            target = target_vec.to(device)
            std = std.to(device)
            mean = mean.to(device)

            pre_seq = target.new_zeros((target.shape[0], target.shape[1], target.shape[2] + 1))
            pre_seq[:, 0:args.n_pre_poses, :-1] = target[:, 0:args.n_pre_poses]
            pre_seq[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints

            if args.model == 'pose_diffusion':
                out_dir_vec = model.sample(pose_dim, pre_seq, in_audio)

            if args.normalize:
                norm_out_dir_vec = out_dir_vec.view(batch_size, -1, 50, 2) * std.unsqueeze(dim=1) + mean.unsqueeze(dim=1)
                norm_out_dir_vec = norm_out_dir_vec.view(batch_size, -1, 50*2)
                out_dir_vec = norm_out_dir_vec

            real_recon, real_feat = net(target)
            gen_recon, gen_feat = net(out_dir_vec)
            real_feat_list.append(real_feat.data.cpu().numpy())
            generated_feat_list.append(gen_feat.data.cpu().numpy())

            rec_loss_real = F.l1_loss(real_recon, target, reduction='mean')
            cos_loss_real = torch.sum(1 - torch.cosine_similarity(real_recon.view(real_recon.shape[0], real_recon.shape[1], -1, 2), target.view(target.shape[0], target.shape[1], -1, 2), dim=-1))

            rec_loss_fake = F.l1_loss(gen_recon, out_dir_vec, reduction='mean')
            cos_loss_fake = torch.sum(1 - torch.cosine_similarity(gen_recon.view(gen_recon.shape[0], gen_recon.shape[1], -1, 2), out_dir_vec.view(out_dir_vec.shape[0], out_dir_vec.shape[1], -1, 2), dim=-1))

            recon_err_diff.append(rec_loss_fake - rec_loss_real)
            cos_err_diff.append(cos_loss_fake - cos_loss_real)

            gen_vids = gen_video(out_dir_vec, source_image_paths, kp_detector, dense_motion_network, inpainting_network)
            tar_vids = gen_video(target_vec, source_image_paths, kp_detector, dense_motion_network, inpainting_network)
            evaluator.push_samples(gen_vids, tar_vids)

            velocity = out_dir_vec[:, 1:, :] - out_dir_vec[:, :-1]
            velocity = torch.cat([torch.zeros(batch_size, 1, velocity.shape[-1]).to(device), velocity], dim=1)
            velocity_diff = velocity[:, 1:, :] - velocity[:, :-1, :]
            velocity_diff = torch.cat([torch.zeros(batch_size, 1, velocity_diff.shape[-1]).to(device), velocity_diff], dim=1)

            accel = torch.mean(torch.abs(velocity_diff), dim=-1)

            thres = 0.0001

            for b in range(batch_size):
                bc_i += 1
                motion_beat_time = []
                for t in range(2, 33):
                    if accel[b][t] < accel[b][t-1] and accel[b][t] < accel[b][t + 1]:
                        if accel[b][t-1] - accel[b][t] >= thres or accel[b][t + 1] - accel[b][t] >= thres:
                            motion_beat_time.append(float(t) / 25.0)
                if (len(motion_beat_time) == 0):
                    continue
                audio = in_audio[b].cpu().numpy()
                audio_beat_time = librosa.onset.onset_detect(y=audio, sr=args.audio_sampling_rate, units='time')
                sum = 0
                for a in audio_beat_time:
                    sum += np.power(math.e, -np.min(np.power((a - motion_beat_time), 2)) / (2 * sigma * sigma))
                bc = sum / len(audio_beat_time)
                bcs += bc

    feat1 = np.vstack(generated_feat_list[:250])
    random_idx = torch.randperm(len(generated_feat_list))[:250]
    shuffle_list = [generated_feat_list[x] for x in random_idx]
    feat2 = np.vstack(shuffle_list)
    feat_dist = np.mean(np.sum(np.absolute(feat1 - feat2), axis=-1))
    diversity_score = feat_dist

    generated_feats = np.vstack(generated_feat_list)
    real_feats = np.vstack(real_feat_list)

    def frechet_distance(samples_A, samples_B):
        A_mu = np.mean(samples_A, axis=0)
        A_sigma = np.cov(samples_A, rowvar=False)
        B_mu = np.mean(samples_B, axis=0)
        B_sigma = np.cov(samples_B, rowvar=False)
        try:
            frechet_dist = calculate_frechet_distance(A_mu, A_sigma, B_mu, B_sigma)
        except ValueError:
            frechet_dist = 1e+10
        return frechet_dist

    ####################################################################
    # frechet distance
    frechet_dist = frechet_distance(generated_feats, real_feats)

    ####################################################################
    # distance between real and generated samples on the latent feature space
    dists = []
    for i in range(real_feats.shape[0]):
        d = np.sum(np.absolute(real_feats[i] - generated_feats[i]))  # MAE
        dists.append(d)
    feat_dist = np.mean(dists)

    bc_score = bcs / float(bc_i)

    with open('metrics.txt', 'a') as f:
        f.write(f'{args.model} ')
        f.write(f'div: {diversity_score} ')
        f.write(f'fgd: {frechet_dist} ')
        f.write(f'ftd: {feat_dist} ')
        f.write(f'bcs: {bc_score} ')
        f.write('\n')

def gen_video(dir_vec, source_image_paths, kp_detector, dense_motion_network, inpainting_network):
    source_images = io.imread_collection(source_image_paths)
    source_images = torch.tensor(img_as_float32(source_images).transpose((0, 3, 1, 2))).to(device)
    source_kp = kp_detector(source_images)
    images = []
    for i in range(dir_vec.shape[1]):
        driving_kp_out = {
            'fg_kp': torch.tensor(dir_vec).view(dir_vec.shape[0], -1, 50, 2)[:, i, :, :].to(device)
        }
        dense_motion_out = dense_motion_network(source_image=source_images,
                                                kp_driving=driving_kp_out,
                                                kp_source=source_kp, bg_param=None,
                                                dropout_flag=False)
        out = inpainting_network(source_images, dense_motion_out)
        images.append(out['prediction'].data.cpu().numpy().transpose(0, 2, 3, 1))
    images = np.stack(images, axis=1)
    return images

def gen_video_with_all(dir_vec, source_image_paths, kp_detector, dense_motion_network, inpainting_network):
    source_images = io.imread_collection(source_image_paths)
    source_images = torch.tensor(img_as_float32(source_images).transpose((0, 3, 1, 2))).to(device)
    source_kp = kp_detector(source_images)
    images = {
        'optical_flow': [],
        'prediction': [],
        'keys': [],
        'prediction_keys': [],
        'occlusion_maps': [],
        'source': source_images.data.cpu().numpy().transpose((0, 2, 3, 1))
    }

    colormap = plt.get_cmap('gist_rainbow')

    for i in range(dir_vec.shape[1]):
        driving_kp_out = {
            'fg_kp': torch.tensor(dir_vec).view(dir_vec.shape[0], -1, 50, 2)[:, i, :, :].to(device)
        }
        dense_motion_out = dense_motion_network(source_image=source_images,
                                                kp_driving=driving_kp_out,
                                                kp_source=source_kp, bg_param=None,
                                                dropout_flag=False)
        out = inpainting_network(source_images, dense_motion_out)

        pred_out = out['prediction'].data.cpu().numpy().transpose(0, 2, 3, 1)

        maps = []
        for map in out['occlusion_map']:
            occlusion = map.data.cpu().numpy().transpose(0, 2, 3, 1)
            occlusion = np.broadcast_to(occlusion, (occlusion.shape[0], occlusion.shape[1], occlusion.shape[2], 3))
            maps.append(occlusion)

        full_mask = []
        for i in range(out['deformed_source'].shape[1]):
            image = out['deformed_source'][:, i].data.cpu()
            image = F.interpolate(image, size=pred_out.shape[1:3])
            mask = out['contribution_maps'][:, i:(i+1)].data.cpu().repeat(1, 3, 1, 1)
            mask = F.interpolate(mask, size=pred_out.shape[1:3])
            image = np.transpose(image.numpy(), (0, 2, 3, 1))
            mask = np.transpose(mask.numpy(), (0, 2, 3, 1))

            if i != 0:
                color = np.array(colormap((i - 1) / (out['deformed_source'].shape[1] - 1)))[:3]
            else:
                color = np.array((0, 0, 0))

            color = color.reshape((1, 1, 1, 3))

            full_mask.append(mask * color)

        deformed_source = sum(full_mask)

        pred_out = out['prediction'].data.cpu().numpy().transpose(0, 2, 3, 1)
        prediction_out_kp = draw_image_with_kp(pred_out[0], driving_kp_out['fg_kp'].cpu().numpy()[0])
        out_kp_only = draw_image_with_kp(np.zeros_like(pred_out[0]), driving_kp_out['fg_kp'].cpu().numpy()[0])

        images['prediction'].append(pred_out)
        images['occlusion_maps'].append(maps)
        images['optical_flow'].append(deformed_source)
        images['prediction_keys'].append(prediction_out_kp)
        images['keys'].append(out_kp_only)

    return images

def gen_full(out_dir_vec, tar_dir_vec, in_audio, source_image_path, save_dir, video_id, kp_detector, dense_motion_network, inpainting_network):
    source_image = io.imread(source_image_path)
    source_image = torch.tensor(img_as_float32(source_image).transpose((2, 0, 1))).to(device)
    source_image = source_image.unsqueeze(dim=0)

    source_kp = kp_detector(source_image)
    images = []
    kp_images = []
    for i in range(len(out_dir_vec)):
        driving_kp_out = {
            'fg_kp': torch.tensor(out_dir_vec).view(-1, 50, 2)[i].to(device)
        }
        dense_motion_out = dense_motion_network(source_image=source_image,
                                            kp_driving=driving_kp_out,
                                            kp_source=source_kp, bg_param=None,
                                            dropout_flag=False)
        out = inpainting_network(source_image, dense_motion_out)

        prediction_out = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
        prediction_kp = draw_image_with_kp(np.zeros_like(prediction_out), driving_kp_out['fg_kp'].cpu().numpy())

        images.append(prediction_out)
        kp_images.append(prediction_kp)

    save_path = f'{save_dir}/{video_id}'
    print(save_path)
    no_audio_video_path = f'{save_path}_no_audio.mp4'
    imageio.mimsave(no_audio_video_path, [img_as_ubyte(frame) for frame in images], fps=25)

    # show audio
    audio = in_audio.astype(np.float32)
    sr = 26666
    audio_path = f'{save_path}.wav'
    sf.write(audio_path, audio, sr)

    # merge audio and video
    merged_video_path = f'{save_path}.mp4'
    cmd = ['ffmpeg', '-loglevel', 'panic', '-y', '-i', no_audio_video_path, '-i', audio_path, '-strict', '-2',
           merged_video_path]

    subprocess.call(cmd)

    os.remove(audio_path)
    os.remove(no_audio_video_path)

    imageio.mimsave(f'{save_dir}/{video_id}_kp.mp4', [img_as_ubyte(frame) for frame in kp_images], fps=25)


def full_val(test_data_loader, model, args, pose_dim, kp_detector, inpainting_network, dense_motion_network):
    n_save = 50
    inpainting_network.eval()
    kp_detector.eval()
    dense_motion_network.eval()

    save_dir = f'results/{args.model_save_path}/{os.path.basename(test_data_loader.dataset.data_dir)}/frames'
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for iter_idx, data in enumerate(test_data_loader, 0):
            target_vec, in_audio, video_id, start_idx, end_idx, std, mean, source_image_paths = data
            batch_size = target_vec.size(0)

            in_audio = in_audio.to(device)
            target = target_vec.to(device)
            std = std.to(device)
            mean = mean.to(device)

            pre_seq = target.new_zeros((target.shape[0], target.shape[1], target.shape[2] + 1))
            pre_seq[:, 0:args.n_pre_poses, :-1] = target[:, 0:args.n_pre_poses]
            pre_seq[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints

            logging.info(f"generating gestures for batch {iter_idx}")
            if args.model == 'pose_diffusion':
                out_dir_vec = model.sample(pose_dim, pre_seq, in_audio)

            if args.normalize:
                norm_out_dir_vec = out_dir_vec.view(batch_size, -1, 50, 2) * std.unsqueeze(dim=1) + mean.unsqueeze(dim=1)
                norm_out_dir_vec = norm_out_dir_vec.view(batch_size, -1, 50*2)
                out_dir_vec = norm_out_dir_vec.data.cpu().numpy()

            videos = []
            for i in range(0, batch_size, 1):
                logging.info(f"generating videos for batch {iter_idx} - {i}")
                slice = out_dir_vec[i:i+1, :, :]
                start_idx_slice = start_idx[i:i+1]
                vid_slice = video_id[i:i+1]
                vids = gen_video(slice, source_image_paths[i:i+1], kp_detector, dense_motion_network, inpainting_network)

                logging.info(f"saving videos for batch {iter_idx} - {i}")

                for j, vid in enumerate(vids):
                    os.makedirs(f'{save_dir}/{vid_slice[j]}_{start_idx_slice[j]}', exist_ok=True)
                    for f, frame in enumerate(vid):
                        io.imsave(f'{save_dir}/{vid_slice[j]}_{start_idx_slice[j]}/{f:0>4}.png', img_as_ubyte(frame))

def demo_val(test_data_loader, model, args, pose_dim, kp_detector, inpainting_network, dense_motion_network):
    inpainting_network.eval()
    kp_detector.eval()
    dense_motion_network.eval()
    save_dir = f'demo_results'
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for iter_idx, data in enumerate(test_data_loader, 0):
            target_vec, in_audio, video_id, start_idx, end_idx, std, mean, source_image_paths = data
            batch_size = target_vec.size(0)

            in_audio = in_audio.to(device)
            target = target_vec.to(device)
            std = std.to(device)
            mean = mean.to(device)

            pre_seq = target.new_zeros((target.shape[0], target.shape[1], target.shape[2] + 1))
            pre_seq[:, 0:args.n_pre_poses, :-1] = target[:, 0:args.n_pre_poses]
            pre_seq[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints

            logging.info(f"generating gestures for batch {iter_idx}")
            if args.model == 'pose_diffusion':
                traj = model.sample_traj(pose_dim, pre_seq, in_audio)

            for traj_index in traj:
                logging.info(f"generating video for {traj_index}")
                out_dir_vec = traj[traj_index].to(device)
                # out_dir_vec = traj
                if args.normalize:
                    norm_out_dir_vec = out_dir_vec.view(batch_size, -1, 50, 2) * std.unsqueeze(dim=1) + mean.unsqueeze(dim=1)
                    norm_out_dir_vec = norm_out_dir_vec.view(batch_size, -1, 50*2)
                    out_dir_vec = norm_out_dir_vec.data.cpu().numpy()

                slice = out_dir_vec
                start_idx_slice = start_idx
                vid_slice = video_id
                images = gen_video_with_all(slice, source_image_paths, kp_detector, dense_motion_network, inpainting_network)

                logging.info(f"saving videos for {traj_index}")

                os.makedirs(f'{save_dir}/{vid_slice[0]}_{start_idx_slice[0]}/{traj_index:0>3}', exist_ok=True)

                try:
                    io.imsave(f'{save_dir}/{vid_slice[0]}_{start_idx_slice[0]}/{traj_index:0>3}/source.png', img_as_ubyte(images['source']))
                    for f, frame in enumerate(images['prediction']):
                        io.imsave(f'{save_dir}/{vid_slice[0]}_{start_idx_slice[0]}/{traj_index:0>3}/prediction_{f:0>4}.png', img_as_ubyte(frame))
                    for f, frame in enumerate(images['optical_flow']):
                        io.imsave(f'{save_dir}/{vid_slice[0]}_{start_idx_slice[0]}/{traj_index:0>3}/optical_flow_{f:0>4}.png', img_as_ubyte(frame))
                    for f, frame in enumerate(images['keys']):
                        io.imsave(f'{save_dir}/{vid_slice[0]}_{start_idx_slice[0]}/{traj_index:0>3}/keys_{f:0>4}.png', img_as_ubyte(frame))
                    for f, frame in enumerate(images['prediction_keys']):
                        io.imsave(f'{save_dir}/{vid_slice[0]}_{start_idx_slice[0]}/{traj_index:0>3}/predkeys_{f:0>4}.png', img_as_ubyte(frame))
                    for f, frame in enumerate(images['occlusion_maps']):
                        for m in frame:
                            io.imsave(f'{save_dir}/{vid_slice[0]}_{start_idx_slice[0]}/{traj_index:0>3}/occlusion_maps_{m.shape[2]}_{f:0>4}.png', img_as_ubyte(m))
                except Exception as e:
                    print(e)
                    continue

def load_clip(file, keypoints_dir, audio_dir, n_poses, subdivision_stride, pose_resampling_fps, audio_sample_length, audio_sample_rate, normalize):
    vid_id, ext = os.path.splitext(file)
    keypoint_file = os.path.join(keypoints_dir, file)# + ext)
    audio_file = os.path.join(audio_dir, vid_id + '.mp3')
    audio, sr = librosa.load(audio_file, sr=audio_sample_rate)
    keypoints = torch.load(keypoint_file, map_location=torch.device('cpu'))

    num_subdivisions = math.floor((len(keypoints) - n_poses) / subdivision_stride) + 1

    sample_keypoints_list = []
    sample_audio_list = []
    video_id_list = []
    start_idx_list = []
    end_idx_list = []
    std = torch.load('std.pt')
    mean = torch.load('mean.pt')

    keypoints = keypoints[:(num_subdivisions * subdivision_stride), :]

    for i in range(num_subdivisions):
        start_idx = i * subdivision_stride
        end_idx = start_idx + n_poses

        sample_keypoints = keypoints[start_idx:end_idx]

        if normalize:
            sample_keypoints = (sample_keypoints - mean) / std

        audio_start = math.floor(start_idx / len(keypoints) * len(audio))
        audio_end = audio_start + audio_sample_length

        if audio_end > len(audio):
            n_padding = audio_end - len(audio)
            padded_data = np.pad(audio, (0, n_padding), mode='symmetric')
            sample_audio = padded_data[audio_start:audio_end]
        else:
            sample_audio = audio[audio_start:audio_end]

        sample_audio_list.append(sample_audio)
        sample_keypoints_list.append(sample_keypoints.view(-1, 50*2))
        video_id_list.append(vid_id)
        start_idx_list.append(start_idx)
        end_idx_list.append(end_idx)

    return sample_keypoints_list, sample_audio_list, \
        video_id_list, start_idx_list, end_idx_list, \
        keypoints, std, mean, audio


def long_full_val(test_data_path, model, args, pose_dim, kp_detector, inpainting_network, dense_motion_network,
                  n_poses,
                  n_pre_poses,
                  subdivision_stride,
                  pose_resampling_fps,
                  audio_sampling_rate,
                  normalize=False):

    save_dir = f'{args.model_save_path.split("/")[1]}'
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    audio_sample_rate = audio_sampling_rate

    audio_sample_length = int(n_poses / pose_resampling_fps * audio_sample_rate)

    logging.info("Reading data '{}'...".format(test_data_path))
    keypoints_dir = os.path.join(test_data_path, 'keypoints')
    audio_dir = os.path.join(test_data_path, 'audio')
    # iters = 50
    # count = 0
    for file in os.listdir(keypoints_dir):
        # if count > iters:
        #     break
        # count += 1

        sample_keypoints_list, sample_audio_list, video_id_list, \
            start_idx_list, end_idx_list, tar_dir_vec, std, mean, audio = load_clip(file, keypoints_dir,
                                                                             audio_dir, n_poses, subdivision_stride,
                                                                             pose_resampling_fps, audio_sample_length,
                                                                             audio_sample_rate, normalize)

        out_list = []

        n_frames = n_poses
        clip_length = audio_sample_length / audio_sample_rate

        seed_seq = sample_keypoints_list[0]
        target_poses = seed_seq
        # pre seq
        pre_seq = torch.zeros((1, target_poses.shape[0], target_poses.shape[1] + 1))
        if seed_seq is not None:
            pre_seq[0, 0:n_pre_poses, :-1] = torch.Tensor(seed_seq[0:n_pre_poses])
            pre_seq[0, 0:n_pre_poses, -1] = 1  # indicating bit for seed poses

        num_subdivision = len(sample_keypoints_list)

        out_dir_vec = None
        for i in range(0, num_subdivision):
            pre_seq = pre_seq.to(device)
            in_audio = torch.from_numpy(sample_audio_list[i]).unsqueeze(0).to(device).float()
            std = std.to(device)
            mean = mean.to(device)

            # prepare pre seq
            if i > 0:
                pre_seq[0, 0:args.n_pre_poses, :-1] = out_dir_vec.squeeze(0)[-args.n_pre_poses:]
                pre_seq[0, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
            pre_seq = pre_seq.float().to(device)

            if args.model == 'pose_diffusion':
                out_dir_vec = model.sample(pose_dim, pre_seq, in_audio)

            if normalize:
                norm_out_dir_vec = out_dir_vec.view(1, -1, 50, 2) * std + mean
                norm_out_dir_vec = norm_out_dir_vec.view(1, -1, 50*2)
                out_seq = norm_out_dir_vec[0, :, :].data.cpu().numpy()
            else:
                out_seq = out_dir_vec[0, :, :].data.cpu().numpy()

            # smoothing motion transition
            if len(out_list) > 0:
                last_poses = out_list[-1][-args.n_pre_poses:]
                out_list[-1] = out_list[-1][:-args.n_pre_poses]  # delete last 4 frames

                for j in range(len(last_poses)):
                    n = len(last_poses)
                    prev = last_poses[j]
                    next = out_seq[j]
                    out_seq[j] = prev * (n - j) / (n + 1) + next * (j + 1) / (n + 1)

            out_list.append(out_seq)

        print('generation took {:.2} s'.format((time.time() - start) / num_subdivision))
        out_dir_vec = tar_dir_vec.cpu().numpy()
        # aggregate results
        out_dir_vec = np.vstack(out_list)

        min_size = min(out_dir_vec.shape[0], tar_dir_vec.shape[0])
        tar_dir_vec = tar_dir_vec[:min_size].cpu().numpy()
        out_dir_vec = out_dir_vec[:min_size]

        in_audio = audio
        source_image_path = f'{test_data_path}/images/{video_id_list[0]}/{start_idx_list[0]:0>4}.png'
        gen_full(out_dir_vec, tar_dir_vec, in_audio, source_image_path, save_dir, video_id_list[0],
                 kp_detector, dense_motion_network, inpainting_network)


def draw_image_with_kp(image, kp_array):
    colormap = plt.get_cmap('gist_rainbow')
    image = np.copy(image)
    spatial_size = np.array(image.shape[:2][::-1])[np.newaxis]
    kp_array = spatial_size * (kp_array + 1) / 2
    num_kp = kp_array.shape[0]
    for kp_ind, kp in enumerate(kp_array):
        rr, cc = disk((kp[1], kp[0]), 5, shape=image.shape[:2])
        image[rr, cc] = np.array(colormap(kp_ind / num_kp))[:3]
    return image


def create_video(save_path, iter_idx, prefix, target, output):
    images = []
    for t, o in zip(target, output):
        tar = torch.tensor(t).view(50, 2).numpy()
        out = torch.tensor(o).view(50, 2).numpy()

        tar_img = draw_image_with_kp(np.zeros([384, 384, 3]), tar)
        out_img = draw_image_with_kp(np.zeros([384, 384, 3]), out)
        images.append(np.concatenate((out_img, tar_img), axis=-2))

    imageio.mimsave(f'results_2/{iter_idx:0>4}.mp4', [img_as_ubyte(frame) for frame in images], fps=1)


def main(mode, checkpoint_path, data_path):
    args, model, lang_model, speaker_model, pose_dim = load_checkpoint_and_model(
        checkpoint_path, device)

    # random seed
    if args.random_seed >= 0:
        set_random_seed(args.random_seed)

    # set logger
    set_logger(args.model_save_path, os.path.basename(__file__).replace('.py', '.log'))

    logging.info("PyTorch version: {}".format(torch.__version__))
    logging.info("CUDA version: {}".format(torch.version.cuda))
    logging.info("{} GPUs, default {}".format(torch.cuda.device_count(), device))

    # load tpsm networks
    with open('ted.yaml') as f:
        config = yaml.safe_load(f)

    inpainting_network = InpaintingNetwork(**config['model_params']['generator_params'],
                                           **config['model_params']['common_params'])
    kp_detector = KPDetector(**config['model_params']['common_params'])
    dense_motion_network = DenseMotionNetwork(**config['model_params']['common_params'],
                                              **config['model_params']['dense_motion_params'])

    tps_checkpoint = torch.load('ted.pth.tar')
    inpainting_network.load_state_dict(tps_checkpoint['inpainting_network'])
    kp_detector.load_state_dict(tps_checkpoint['kp_detector'])
    dense_motion_network.load_state_dict(tps_checkpoint['dense_motion_network'])

    inpainting_network.to(device)
    kp_detector.to(device)
    dense_motion_network.to(device)

    collate_fn = default_collate_fn

    def load_dataset(path):
        dataset = SpeechMotionDataset(path,
                                      n_poses=args.n_poses,
                                      subdivision_stride=34,#args.subdivision_stride,
                                      pose_resampling_fps=args.motion_resampling_framerate,
                                      audio_sampling_rate=args.audio_sampling_rate,
                                      normalize=args.normalize
                                      )
        print(len(dataset))
        return dataset

    if mode == 'eval':
        val_data_path = data_path
        eval_net_path = 'output/TED_Expressive_output/AE-cos1e-3/checkpoint_best.bin'
        evaluator = Evaluator(args, eval_net_path, lang_model, device)
        val_dataset = load_dataset(val_data_path)
        data_loader = DataLoader(dataset=val_dataset, batch_size=8, collate_fn=collate_fn,
                                 shuffle=False, drop_last=True, num_workers=args.loader_workers)
        evaluate(data_loader, model, evaluator, args, pose_dim,
                         kp_detector, inpainting_network, dense_motion_network)

    elif mode == 'short':
        val_dataset = load_dataset(data_path)
        data_loader = DataLoader(dataset=val_dataset, batch_size=1, collate_fn=collate_fn, #args.batch_size
                                 shuffle=False, drop_last=True, num_workers=args.loader_workers)
        val_dataset.set_lang_model(lang_model)

        full_val(data_loader, model, args, pose_dim, kp_detector,
                 inpainting_network, dense_motion_network,)

    elif mode == "long":
        long_full_val(data_path, model, args, pose_dim, kp_detector,
                      inpainting_network, dense_motion_network,
                      n_poses=args.n_poses,
                      n_pre_poses=args.n_pre_poses,
                      subdivision_stride=args.n_poses - args.n_pre_poses,
                      pose_resampling_fps=args.motion_resampling_framerate,
                      audio_sampling_rate=args.audio_sampling_rate,
                      normalize=args.normalize)
    elif mode == "demo":
        val_dataset = load_dataset(data_path)
        data_loader = DataLoader(dataset=val_dataset, batch_size=1, collate_fn=collate_fn,
                                 shuffle=False, drop_last=True, num_workers=args.loader_workers)
        val_dataset.set_lang_model(lang_model)

        demo_val(data_loader, model, args, pose_dim, kp_detector,
                 inpainting_network, dense_motion_network)
    else:
        assert False, 'wrong mode'


if __name__ == '__main__':
    mode = sys.argv[1]
    checkpoint = sys.argv[2]
    data_path = sys.argv[3]

    assert mode in ["eval", "short", "long", "demo"]

    ckpt_path = f'output/{checkpoint}/pose_diffusion_checkpoint_499.bin'

    main(mode, ckpt_path, data_path)
