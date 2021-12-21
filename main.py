import os
import json
import argparse
import random
import numpy as np
import pandas as pd
import torch
import agents


def train(args):
    agent = getattr(agents, args.agent)(args)
    path = agent.logger.log_dir

    # save current config to log directory
    with open(os.path.join(path, 'config.json'), 'w') as f:
        json.dump(args.__dict__, f, default=str)

    for idx in range(args.train_iter):
        # train agent
        if idx % args.eval_every == 0:
            agent.eval()
        agent.train()

        # save model
        if idx % args.save_step == 0:
            torch.save(
                agent.model.state_dict(),
                os.path.join(path, 'model.pt')
            )


def test(args):
    agent = getattr(agents, args.agent)(args)
    _, ckpt_name = os.path.split(args.checkpoint)
    path = os.path.join(agent.logger.log_dir, '{}.csv'.format(ckpt_name))

    # test run
    pnl = agent.test()

    # match datetime index from the environment
    agent.env.test()
    index = agent.env.prices.index[-len(pnl):]

    # save series to csv
    pd.Series(pnl, index=index).to_csv(path)


def visualize(args):
    assert os.path.isdir(args.checkpoint), "must provide directory by the checkpoint argument"

    import matplotlib.pyplot as plt
    import seaborn as sns
    from mpt import portfolio_return 

    sns.set_style('whitegrid')

    # load test data
    dfs = []
    for filename in os.listdir(args.checkpoint):
        if filename.endswith('.csv'):
            path = os.path.join(args.checkpoint, filename)
            df = pd.read_csv(path, index_col='Date')
            df.columns = [filename[:-4]]
            dfs.append(df)
    dfs = pd.concat(dfs, axis=1)

    # add DJIA
    djia = pd.read_csv('data/^DJI.csv', index_col='Date').Close
    djia = djia[dfs.index]
    djia = djia / djia.iloc[0] - 1.0
    dfs['DJIA'] = djia

    # add MPT benchmarks
    dfs.index = pd.to_datetime(dfs.index)
    benchmark = portfolio_return(args, method='max_sharpe')
    benchmark = benchmark[dfs.index]
    benchmark = benchmark / benchmark.iloc[0] - 1.0
    dfs['max_sharpe'] = benchmark

    benchmark = portfolio_return(args, method='min_volatility')
    benchmark = benchmark[dfs.index]
    benchmark = benchmark / benchmark.iloc[0] - 1.0
    dfs['min_volatility'] = benchmark

    dfs.plot(figsize=(20, 10))
    plt.legend(loc='upper left')
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/result.png')


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
    common.add_argument("mode", type=str, default='test_logger')
    common.add_argument("--tag", type=str, default='')
    common.add_argument("--seed", type=int, default=-1)
    common.add_argument("--config", type=str, default=None)

    log = parser.add_argument_group("logging options")
    log.add_argument("--log_level", type=int, default=20)
    log.add_argument("--log_step", type=int, default=10000)
    log.add_argument("--save_step", type=int, default=1000000)
    log.add_argument("--debug", "-d", action="store_true")
    log.add_argument("--quiet", "-q", action="store_true")

    dirs = parser.add_argument_group("directory configurations")
    dirs.add_argument("--log_dir", type=str, default='logs')
    dirs.add_argument("--data_dir", type=str, default='data')
    dirs.add_argument("--checkpoint", type=str, default=None)

    env = parser.add_argument_group("environment configurations")
    env.add_argument("--env", type=str.lower, default='djia_new')
    env.add_argument("--start_train", type=str, default="2009-01-01")
    env.add_argument("--start_val", type=str, default="2018-01-01")
    env.add_argument("--start_test", type=str, default="2019-12-01")
    env.add_argument("--initial_balance", type=float, default=1e6)
    env.add_argument("--transaction_cost", type=float, default=1e-3)

    training = parser.add_argument_group("training configurations")
    training.add_argument("--agent", type=str.lower, default='ddpg')
    training.add_argument("--arch", type=str.lower, default='cnn',
                          choices=['cnn', 'transformer'])
    training.add_argument("--train_iter", type=int, default=100000000)
    training.add_argument("--eval_every", type=int, default=10000)
    training.add_argument("--update_every", type=int, default=128)
    training.add_argument("--update_epoch", type=int, default=4)
    training.add_argument("--buffer_size", type=int, default=50000)
    training.add_argument("--warmup", type=int, default=1000)
    training.add_argument("--batch_size", type=int, default=32)

    training.add_argument("--lr_critic", type=float, default=1e-4)
    training.add_argument('-lr', "--lr_actor", type=float, default=1e-4)
    training.add_argument("--grad_clip", type=float, default=0.5)
    training.add_argument("--sigma", type=float, default=0.1)
    training.add_argument("--gamma", type=float, default=0.99)
    training.add_argument("--lambda", type=float, default=0.95)
    training.add_argument("--polyak", type=float, default=0.99)
    training.add_argument("--cr_coef", type=float, default=0.5)
    training.add_argument("--ent_coef", type=float, default=0.0)
    training.add_argument("--cliprange", type=float, default=0.1)

    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, 'r') as f:
            args.__dict__ = json.load(f)

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
