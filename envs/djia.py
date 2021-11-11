from envs.base import Environment
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class DJIA(Environment):
    def __init__(self, args=None):
        self.args = args

        # predefined constants
        # modified from https://github.com/AI4Finance-Foundation/FinRL
        self._balance_scale = 1e-4
        self._price_scale = 1e-2
        self._reward_scale = 1e-4
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
        self.all_prices = prices

        # default to training
        self.train()

        # initialize environment
        _ = self.reset()

    def train(self):
        start = self.args.start_train
        end = self.args.start_val - timedelta(days=1)
        self.prices = self.all_prices[start:end]

    def eval(self):
        start = self.args.start_val
        end = self.args.start_test - timedelta(days=1)
        self.prices = self.all_prices[start:end]

    def test(self):
        start = self.args.start_test
        self.prices = self.all_prices[start:]

    @property
    def observation_space(self):
        return (61,)

    @property
    def action_space(self):
        # actions are assumed to be constrained to [-1.0, 1.0]
        return (30,)

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
            profit = self.total_asset / self.args.initial_balance - 1.0
            state = self.reset()
            return state, reward, True, {'profit': profit}

        # create state vector
        p = prices * self._price_scale
        h = self.holdings * self._price_scale
        b = max(self.balance, 1e4)  # cutoff value defined in FinRL
        b *= np.ones(1) * self._balance_scale
        state = np.concatenate([p, h, b], axis=0)
        return state, reward, False, {}

