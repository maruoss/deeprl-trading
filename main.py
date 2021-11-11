import argparse
import random
import numpy as np
import torch
from datetime import datetime


def train(args):
    from agents.ddpg.agent import DDPG

    agent = DDPG(args)
    for idx in range(args.train_iter):
        if idx % args.eval_every == 0:
            agent.eval()
        agent.train()


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
    log.add_argument("--log_step", type=int, default=10000)
    log.add_argument("--save_step", type=int, default=100000)
    log.add_argument("--debug", "-d", action="store_true")
    log.add_argument("--quiet", "-q", action="store_true")

    dirs = parser.add_argument_group("directory configurations")
    dirs.add_argument("--log_dir", type=str, default='logs')
    dirs.add_argument("--data_dir", type=str, default='data')
    dirs.add_argument("--checkpoint", type=str, default=None)

    env = parser.add_argument_group("environment configurations")
    env.add_argument("--env", type=str, default='djia')
    env.add_argument("--start_train", type=str, default="2009-01-01")
    env.add_argument("--start_val", type=str, default="2018-12-01")
    env.add_argument("--start_test", type=str, default="2019-12-01")
    env.add_argument("--initial_balance", type=float, default=1e6)
    env.add_argument("--transaction_cost", type=float, default=1e-3)

    training = parser.add_argument_group("training configurations")
    training.add_argument("--train_iter", type=int, default=1000000)
    training.add_argument("--eval_every", type=int, default=10000)
    training.add_argument("--buffer_size", type=int, default=50000)
    training.add_argument("--warmup", type=int, default=1000)
    training.add_argument("--batch_size", type=int, default=128)
    training.add_argument("--lr_actor", type=float, default=1e-4)
    training.add_argument("--lr_critic", type=float, default=1e-3)
    training.add_argument("--grad_clip", type=float, default=0.01)
    training.add_argument("--sigma", type=float, default=0.1)
    training.add_argument("--gamma", type=float, default=0.99)
    training.add_argument("--polyak", type=float, default=0.99)

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

    # check dates
    args.start_train = datetime.strptime(args.start_train, "%Y-%M-%d")
    args.start_val = datetime.strptime(args.start_val, "%Y-%M-%d")
    args.start_test = datetime.strptime(args.start_test, "%Y-%M-%d")
    assert args.start_train < args.start_val, "the start of training must be earlier than the validation"
    assert args.start_val < args.start_test, "the start of validation must be earlier than the test"

    globals()[args.mode](args)
