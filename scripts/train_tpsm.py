import shutil
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime
import pprint
import sys
import time

from data_loader.tpsm_data_loader import *
from model.pose_diffusion import PoseDiffusion
from parse_args_diffusion import parse_args
from train_eval.train_diffusion import train_iter_diffusion
from utils.average_meter import AverageMeter
import utils.train_utils
from torch.nn.utils import clip_grad_norm_

[sys.path.append(i) for i in ['.', '..']]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def init_model(args, _device):
    # init model
    if args.model == 'pose_diffusion':
        print("init diffusion model")
        model = PoseDiffusion(args).to(_device)

    return model


def train_epochs(args, train_data_loader, lang_model, pose_dim):
    start = time.time()
    loss_meters = [AverageMeter('loss'), AverageMeter('var_loss'), AverageMeter('gen'), AverageMeter('dis'),
                   AverageMeter('KLD'), AverageMeter('DIV_REG'),
                   AverageMeter('noise'), AverageMeter('dist'), AverageMeter('pos')]

    tb_path = args.name + '_' + str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    tb_writer = SummaryWriter(log_dir=str(Path(args.model_save_path).parent / 'tensorboard_runs' / tb_path))

    # interval params
    print_interval = int(len(train_data_loader) / 5)
    save_model_epoch_interval = 50

    model = init_model(args, device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

    # training
    global_iter = 0
    for epoch in range(args.epochs):
        # save model
        if (epoch % save_model_epoch_interval == 0 and epoch > 0) or epoch == args.epochs - 1:
            state_dict = model.state_dict()

            save_name = '{}/{}_checkpoint_{:03d}.bin'.format(args.model_save_path, args.name, epoch)

            utils.train_utils.save_checkpoint({
                'args': args, 'epoch': epoch, 'lang_model': lang_model,
                'pose_dim': pose_dim, 'state_dict': state_dict,
            }, save_name)

        # train iter
        iter_start_time = time.time()
        for iter_idx, data in enumerate(train_data_loader, 0):
            global_iter += 1
            target_vec, in_audio, video_id, start_idx, end_idx, std, mean, _ = data

            batch_size = target_vec.size(0)
            in_audio = in_audio.to(device)
            target_vec = target_vec.to(device)
            std = std.to(device)
            mean = mean.to(device)

            # train
            # make pre seq input
            pre_seq = target_vec.new_zeros((target_vec.shape[0], target_vec.shape[1], target_vec.shape[2] + 1))
            pre_seq[:, 0:args.n_pre_poses, :-1] = target_vec[:, 0:args.n_pre_poses]
            pre_seq[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints

            optimizer.zero_grad()
            model.train()

            losses = model.get_loss(target_vec, pre_seq, in_audio, std, mean)

            loss = sum(val for val in losses.values())
            loss.backward()
            clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

            loss_dict = {
                'loss': loss.item()
            }

            for k in losses:
                loss_dict[k] = losses[k].item()

            # loss values
            for loss_meter in loss_meters:
                name = loss_meter.name
                if name in loss_dict:
                    loss_meter.update(loss_dict[name], batch_size)

            # write to tensorboard
            for key in loss_dict.keys():
                tb_writer.add_scalar(key + '/train', loss_dict[key], global_iter)

            # print training status
            if (iter_idx + 1) % print_interval == 0:
                print_summary = 'EP {} ({:3d}) | {:>8s}, {:.0f} samples/s | '.format(
                    epoch, iter_idx + 1, utils.train_utils.time_since(start),
                           batch_size / (time.time() - iter_start_time))
                for loss_meter in loss_meters:
                    if loss_meter.count > 0:
                        print_summary += '{}: {:.3f}, '.format(loss_meter.name, loss_meter.avg)
                        loss_meter.reset()
                logging.info(print_summary)

            iter_start_time = time.time()

    tb_writer.close()


def main(config):
    args = config['args']

    # random seed
    if args.random_seed >= 0:
        utils.train_utils.set_random_seed(args.random_seed)

    # set logger
    utils.train_utils.set_logger(args.model_save_path, os.path.basename(__file__).replace('.py', '.log'))

    logging.info("PyTorch version: {}".format(torch.__version__))
    logging.info("CUDA version: {}".format(torch.version.cuda))
    logging.info("{} GPUs, default {}".format(torch.cuda.device_count(), device))
    logging.info(pprint.pformat(vars(args)))

    collate_fn = default_collate_fn

    # dataset
    train_dataset = SpeechMotionDataset(args.train_data_path[0],
                                        n_poses=args.n_poses,
                                        subdivision_stride=args.subdivision_stride,
                                        pose_resampling_fps=args.motion_resampling_framerate,
                                        audio_sampling_rate=args.audio_sampling_rate,
                                        normalize=args.normalize,
                                        )
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=args.loader_workers, pin_memory=True,
                              collate_fn=collate_fn
                              )

    # train
    pose_dim = args.pose_dim
    train_epochs(args, train_loader, lang_model, pose_dim=pose_dim)


if __name__ == '__main__':
    _args = parse_args()
    main({'args': _args})
