import configargparse


def str2bool(v):
    """ from https://stackoverflow.com/a/43357954/1361529 """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = configargparse.ArgParser()
    parser.add('-c', '--config', required=True, is_config_file=True, help='Config file path')
    parser.add("--name", type=str, default="main")
    parser.add("--train_data_path", action="append")
    parser.add("--val_data_path", action="append")
    parser.add("--test_data_path", action="append")
    parser.add("--model_save_path", required=True)
    parser.add("--pose_representation", type=str, default='3d_vec')
    parser.add("--random_seed", type=int, default=-1)
    parser.add("--save_result_video", type=str2bool, default=True)

    # model
    parser.add("--model", type=str, required=True)
    parser.add("--epochs", type=int, default=10)
    parser.add("--batch_size", type=int, default=50)
    parser.add("--hidden_size", type=int, default=200)
    parser.add("--input_context", type=str, default='audio')

    # dataset
    parser.add("--motion_resampling_framerate", type=int, default=25)
    parser.add("--audio_sampling_rate", type=int, default=26500)
    parser.add("--n_poses", type=int, default=50)
    parser.add("--n_pre_poses", type=int, default=5)
    parser.add("--subdivision_stride", type=int, default=5)
    parser.add("--loader_workers", type=int, default=0)

    parser.add("--pose_dim", type=int, required=True)
    parser.add("--latent_dim", type=int, default=128)

    parser.add("--diff_hidden_dim", type=int, required=True)
    parser.add("--block_depth", type=int, default=8)

    parser.add("--normalize", type=str2bool, default=False)
    parser.add("--loss_noise", type=str2bool, default=True)
    parser.add("--loss_dist", type=str2bool, default=True)
    parser.add("--loss_pos", type=str2bool, default=True)
    parser.add("--k", type=int, default=49)

    # training
    parser.add("--learning_rate", type=float, default=0.001)

    parser.add("--classifier_free", type=str2bool, default=False)
    parser.add("--null_cond_prob", type=float, default=None)

    # eval
    parser.add("--eval_net_path", type=str, default='')

    # testing
    parser.add("--mode", type=str, default='full')
    parser.add("--checkpoint", type=str, default='')

    args = parser.parse_args()
    return args
