import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import torch
import sys

# if there's overlap between args_list and commandline input, use commandline input
def solve_argv_conflict(args_list):
    arguments_to_be_removed = []
    arguments_size = []

    for argv in sys.argv[1:]:
        if argv.startswith('-'):
            size_count = 1
            for i, args in enumerate(args_list):
                if args == argv:
                    arguments_to_be_removed.append(args)
                    for more_args in args_list[i+1:]:
                        if not more_args.startswith('-'):
                            size_count += 1
                        else:
                            break
                    arguments_size.append(size_count)
                    break

    for args, size in zip(arguments_to_be_removed, arguments_size):
        args_index = args_list.index(args)
        for _ in range(size):
            args_list.pop(args_index)

from datetime import datetime

def get_time_stamp():
    now = datetime.now()
    year = now.strftime('%Y')
    month = now.strftime('%m')
    day = now.strftime('%d')
    hour = now.strftime('%H')
    minute = now.strftime('%M')
    second = now.strftime('%S')
    return '{}-{}-{}-{}-{}-{}'.format(month, day, year, hour, minute, second)

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def print_error(*message):
    print('\033[91m', 'ERROR ', *message, '\033[0m')
    raise RuntimeError


def print_ok(*message):
    print('\033[92m', *message, '\033[0m')


def print_warning(*message):
    print('\033[91m', *message, '\033[0m')


def print_info(*message):
    print('\033[96m', *message, '\033[0m')


def print_white(*message):
    print('\033[37m', *message, '\033[0m')


def pathlib_file(file_name):
    if isinstance(file_name, str):
        file_name = Path(file_name)
    elif not isinstance(file_name, Path):
        raise TypeError(f'Please check the type of '
                        f'the filename:{file_name}')
    return file_name


def save_to_json(data, file_name):
    file_name = pathlib_file(file_name)
    if not file_name.parent.exists():
        Path.mkdir(file_name.parent, parents=True)
    with file_name.open('w') as f:
        json.dump(data, f, indent=2)


def load_from_json(file_name):
    file_name = pathlib_file(file_name)
    with file_name.open('r') as f:
        data = json.load(f)
    return data

def plot_loss(losses, labels, title=None, font_size=20, subsample=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.tick_params(axis='x', labelsize=font_size)
    ax.xaxis.get_offset_text().set_fontsize(font_size)
    ax.yaxis.get_offset_text().set_fontsize(font_size)
    ax.xaxis.set_tick_params(labelsize=font_size)
    ax.yaxis.set_tick_params(labelsize=font_size)
    for loss, label in zip(losses, labels):
        if len(loss.shape) > 1:
            x = loss[:, 0]
            y = loss[:, 1]
        else:
            x = np.arange(len(loss))
            y = loss
        if subsample is not None:
            x = x[::subsample]
            y = y[::subsample]
        ax.plot(x, y, label=label)
    ax.set_xlabel('No. of iterations', fontsize=font_size)
    ax.set_ylabel('Loss', fontsize=font_size)
    if title is not None:
        ax.set_title(title, fontsize=font_size)
    ax.legend(fontsize=font_size)
    return fig, ax

def get_grad_free_parser_args():
    parser = argparse.ArgumentParser('')
    parser.add_argument("--model", type=str, default='abstract_finger_new')
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--save-dir', type=str, default='data')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--record-file-name', type=str, default='grad_free')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--optim', '-o', choices=['TwoPointsDE', 'NGOpt',
                                                  'OnePlusOne', 'CMA', 'TBPSA',
                                                  'PSO', 'RandomSearch', 'DiagonalCMA', 'FCMA'],
                        default='OnePlusOne')
    parser.add_argument('--no-design-optim', action='store_true', help = 'whether control-only')
    parser.add_argument('--max-iters', type=int, default=5000)
    parser.add_argument('--popsize', type=int, default=None)
    parser.add_argument('--single_stage', action='store_true')
    parser.add_argument('--load-dir', type = str, default = None, help = 'load optimized parameters')
    parser.add_argument('--visualize', type=str, default='True', help = 'whether visualize the simulation')
    parser.add_argument('--verbose', default = False, action = 'store_true', help = 'verbose output')
    args = parser.parse_args()
    set_random_seed(args.seed)
    args.save_dir = Path(args.save_dir)
    if args.single_stage:
        args.save_dir = args.save_dir.joinpath('single_stage')
    args.save_dir = args.save_dir.joinpath(f'{args.optim}')
    if 'CMA' in args.optim and args.popsize is not None:
        args.save_dir = args.save_dir.joinpath(f'popsize_{args.popsize}')

    args.save_dir = args.save_dir.joinpath(f'seed_{args.seed}')
    return args

def visualize_tactile_image(tactile_array, space = 30, shear_threshold = 0.00015, normal_threshold = 0.0008):
    space = 30
    T = len(tactile_array)
    nrows = tactile_array.shape[0]
    ncols = tactile_array.shape[1]

    imgs_tactile = np.zeros((nrows * space, ncols * space, 3), dtype = float)

    for row in range(nrows):
        for col in range(ncols):
            loc0_x = row * space + space // 2
            loc0_y = col * space + space // 2
            loc1_x = loc0_x + tactile_array[row, col][0] / shear_threshold * space
            loc1_y = loc0_y + tactile_array[row, col][1] / shear_threshold * space
            color = (0., max(0., 1. + tactile_array[row][col][2] / normal_threshold), min(1., -tactile_array[row][col][2] / normal_threshold))
            
            cv2.arrowedLine(imgs_tactile, (int(loc0_y), int(loc0_x)), (int(loc1_y), int(loc1_x)), color, 6, tipLength = 0.4)
    
    return imgs_tactile

def visualize_depth_image(tactile_forces, threshold = 0.00120):
    img = np.zeros((tactile_forces.shape[0], tactile_forces.shape[1]))
    for i in range(tactile_forces.shape[0]):
        for j in range(tactile_forces.shape[1]):
            img[i][j] = min(1., -tactile_forces[i][j][2] / threshold)
    return img