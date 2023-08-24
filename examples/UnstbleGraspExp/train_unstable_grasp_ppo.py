import sys, os

base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(base_dir)

import yaml
from arguments import *
from utils.common import *

import algorithms.ppo as ppo

if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    args_list = ['--logdir', './trained_models/',
                 '--log-interval', '1',
                 '--save-interval', '25',
                 '--render-interval', '0',
                 '--seed', '0']

    solve_argv_conflict(args_list)
    parser = get_rl_parser()
    args = parser.parse_args(args_list + sys.argv[1:])

    # load config
    args.train = not args.play
    file_cfg = './cfg' if args.train else args.cfg
    with open(file_cfg + '/cfg.yaml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    if not args.no_time_stamp:
        args.logdir = os.path.join(args.logdir, get_time_stamp())

    if args.train:
        vargs = vars(args)
        cfg["params"]["general"] = {}
        for key in vargs.keys():
            cfg["params"]["general"][key] = vargs[key]
    else:
        cfg["params"]["general"]["cfg"] = args.cfg + '/cfg.yaml'
        cfg["params"]["general"]["checkpoint"] = args.cfg + '/models/best_model.pt'
        cfg["params"]["general"]["train"] = False
        cfg["params"]["general"]["render"] = True
        cfg["params"]["general"]["num_games"] = 100

    algo = ppo.PPO(cfg)

    if args.train:
        algo.train()
    else:
        algo.play(cfg)
