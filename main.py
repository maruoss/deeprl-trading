import argparse
import random
import numpy as np
import torch


def test_logger(args):
    from utils.logger import Logger

    # initialize logger
    logger = Logger('test', args=args)
    logger.log("Testing logger functionality...")
    logger.log("Logs saved to {}".format(logger.log_dir))

    # log built-in levels
    logger.log("Logging DEBUG level", lvl='DEBUG')
    logger.log("Logging INFO level", lvl='INFO')
    logger.log("Logging WARNING level", lvl='WARNING')
    logger.log("Logging ERROR level", lvl='ERROR')
    logger.log("Logging CRITICAL level", lvl='CRITICAL')

    # add logging level
    logger.add_level('NEW', 21, color='grey')
    logger.log("Logging NEW level", lvl='NEW')

    # check excepthook
    raise Exception("Checking system excepthook")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Deep Reinforcement Learning applied to Stock Trading - Data Augmentation using Generative Adversarial Networks"
    )
    common = parser.add_argument_group("common configurations")
    common.add_argument("--mode", type=str, default='test_logger')
    common.add_argument("--tag", type=str, default='')
    common.add_argument("--seed", type=int, default=-1)

    log = parser.add_argument_group("logging options")
    log.add_argument("--log_level", type=int, default=20)
    log.add_argument("--log_step", type=int, default=100)
    log.add_argument("--save_step", type=int, default=100)
    log.add_argument("--debug", "-d", action="store_true")
    log.add_argument("--quiet", "-q", action="store_true")

    dirs = parser.add_argument_group("directory configurations")
    dirs.add_argument("--log_dir", type=str, default='logs')
    dirs.add_argument("--data_dir", type=str, default='data')
    dirs.add_argument("--checkpoint", type=str, default=None)

    training = parser.add_argument_group("training options")

    args = parser.parse_args()
    # set random seed
    if args.seed == -1:
        random.seed(None)
        args.seed = random.randrange(0, int(1e4))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # use cuda when available
    if not hasattr(args, 'device'):
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # set logging level
    if args.debug:
        args.log_level = 1
    elif args.quiet:
        args.log_level = 30

    globals()[args.mode](args)
