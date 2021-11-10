from envs.base import Environment
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class DJIA(Environment):
    def __init__(self, args=None, split="train"):
        assert split in ['train', 'val', 'test']
        self.args = args

        # predefined constants
        # referred to https://github.com/AI4Finance-Foundation/FinRL
        self._balance_scale = 2 ** -12
        self._price_scale = 2 ** -6
        self._reward_scale = 2 ** -11
        self._max_stocks = 100
        self._min_action = int(0.1 * self._max_stocks)

        # load stock data
        prices, tickers = [], []
        root = os.path.join(args.data_dir, 'dow30')
        for filename in os.listdir(root):
            if filename.endswith('.csv'):
                tickers.append(filename[:-4])
                path = os.path.join(root, filename)
                df = pd.read_csv(path, index_col='Date')
                prices.append(df.Close)
        prices = pd.concat(prices, axis=1)
        prices.index = pd.to_datetime(prices.index)
        prices.columns = tickers
        prices.sort_index(axis=0, inplace=True)

        # slice dates
        start_train = datetime.strptime(args.start_train, "%Y-%M-%d")
        start_val = datetime.strptime(args.start_val, "%Y-%M-%d")
        start_test = datetime.strptime(args.start_test, "%Y-%M-%d")
        assert start_train < start_val, "the start of training must be earlier than the validation"
        assert start_val < start_test, "the start of validation must be earlier than the test"

        if split == 'train':
            self.prices = prices[start_train:start_val - timedelta(days=1)]
        elif split == 'val':
            self.prices = prices[start_val:start_test - timedelta(days=1)]
        else:
            self.prices = prices[start_test:]

        # initialize environment
        _ = self.reset()

    @property
    def observation_space(self):
        # s = [p, h, b]
        p = np.zeros((2, 30))
        p[1] = np.inf
        h = np.zeros((2, 30))
        h[1] = np.inf
        b = np.zeros((2, 1))
        b[1] = np.inf
        return np.concatenate([p, h, b], axis=1)

    @property
    def action_space(self):
        # actions constrained to [-1.0, 1.0]
        acs = np.zeros((2, 30))
        acs[0] = -1.0
        acs[1] = 1.0
        return acs

    def reset(self):
        self.head = 0
        self.balance = self.args.initial_balance
        self.holdings = np.zeros(30)
        self.total_asset = self.balance
        self.total_reward = 0.0

        p = self.prices.iloc[self.head].values * self._price_scale
        h = self.holdings * self._price_scale
        b = max(self.balance, 1e4)  # cutoff value defined in FinRL
        b *= np.ones(1) * self._balance_scale
        return np.concatenate([p, h, b], axis=0)

    def step(self, action):
        # rescale actions
        action = (action * self._max_stocks).astype(int)

        # update prices and holdings
        self.head += 1
        if self.head >= len(self.prices):
            raise KeyError("environment must be reset")

        prices = self.prices.iloc[self.head].values
        tc = self.args.transaction_cost
        # sells
        for idx in np.where(action < -self._min_action)[0]:
            shares = min(-action[idx], self.holdings[idx])
            self.holdings[idx] -= shares
            self.balance += prices[idx] * shares * (1 - tc)
        # buys
        for idx in np.where(action > self._min_action)[0]:
            shares = min(action[idx], self.balance // prices[idx])
            self.holdings[idx] += shares
            self.balance -= prices[idx] * shares * (1 + tc)

        # calculate asset gains
        total_asset = self.balance + (prices * self.holdings).sum()
        reward = (total_asset - self.total_asset) * self._reward_scale
        self.total_asset = total_asset
        self.total_reward = self.args.gamma * self.total_reward + reward

        # check if at terminal state
        if self.head == len(self.prices) - 1:
            reward = self.total_reward
            state = self.reset()
            return state, reward, True

        # create state vector
        p = prices * self._price_scale
        h = self.holdings * self._price_scale
        b = max(self.balance, 1e4)  # cutoff value defined in FinRL
        b *= np.ones(1) * self._balance_scale
        state = np.concatenate([p, h, b], axis=0)
        return state, reward, False
