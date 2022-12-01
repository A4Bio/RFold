import argparse


def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--display_step', default=10, type=int, help='Interval in batches between display of training metrics')
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--ex_name', default='Debug', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=111, type=int)

    # dataset parameters
    parser.add_argument('--data_name', default='ArchiveII', choices=['ArchiveII', 'RNAStralign', 'bpRNA'])
    parser.add_argument('--data_root', default='./data/archiveII_all')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=8, type=int)

    # Training parameters
    parser.add_argument('--epoch', default=1, type=int, help='end epoch')
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')

    # debug parameters
    parser.add_argument('--num_hidden', default=128, type=int)
    parser.add_argument('--pf_dim', default=128, type=int)
    parser.add_argument('--num_heads', default=2, type=int)
    parser.add_argument('--dropout', default=0.3, type=float)
    return parser.parse_args()