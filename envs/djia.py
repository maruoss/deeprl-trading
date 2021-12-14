from envs.base import Environment
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class DJIA(Environment):
    def __init__(self, args=None):
        self.args = args

        # check dates
        start_train = datetime.strptime(args.start_train, "%Y-%M-%d")
        start_val = datetime.strptime(args.start_val, "%Y-%M-%d")
        start_test = datetime.strptime(args.start_test, "%Y-%M-%d")
        assert start_train < start_val, "the start of training must be earlier than the validation"
        assert start_val < start_test, "the start of validation must be earlier than the test"

        # predefined constants
        # modified from https://github.com/AI4Finance-Foundation/FinRL #TODO: did not find scaling done on each b, p, r
        # self._balance_scale = 1e-4
        # self._price_scale = 1e-2
        # self._reward_scale = 1e-4
        # self._max_stocks = 100
        # self._min_action = int(0.1 * self._max_stocks)
        # self._min_action = 0.1
        self._action_scale = 10.0
        self._reward_scale = 100.0

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
        start = datetime.strptime(self.args.start_train, "%Y-%M-%d")
        end = datetime.strptime(self.args.start_val, "%Y-%M-%d")
        self.prices = self.all_prices[start:end - timedelta(days=1)]

    def eval(self):
        start = datetime.strptime(self.args.start_val, "%Y-%M-%d")
        end = datetime.strptime(self.args.start_test, "%Y-%M-%d")
        self.prices = self.all_prices[start:end - timedelta(days=1)]

    def test(self):
        start = datetime.strptime(self.args.start_test, "%Y-%M-%d")
        self.prices = self.all_prices[start:]

    @property
    def observation_space(self):
        return (31, 26)

    @property
    def action_space(self):
        # actions are assumed to be constrained to [-1.0, 1.0]
        return (31,)

    def reset(self):
        self.head = 25
        self.balance = self.args.initial_balance
        self.holdings = np.zeros(30)
        self.total_asset = self.balance
        # self.total_reward = 0.0
        
        # initialize lagged return matrix
        self.lag_returns = np.zeros([31, 25]) # row 1 is cash balance, with lag returns all 0.
        self.lag_returns[1:, :] = self.prices.iloc[self.head - 25: self.head + 1].pct_change().dropna(axis=0).T.values # assign stocks: [#stocks, #lagged returns]
        
        #initialize weights
        self.weights = np.zeros(31).reshape(-1, 1) # assume sums to 1, no negative weights. shape: [#stocks, 1]

        return np.concatenate([self.lag_returns, self.weights], axis=1) # return shape [31, 26]

    def step(self, action):
        # rescale actions
        # action *= self._action_scale

        # update prices and holdings
        self.head += 1
        if self.head >= len(self.prices):
            raise KeyError("environment must be reset")

        # Take scaled softmax of action vector: R^31 -> [0, 1]
        exp = np.exp(action - action.max())
        action = (exp / exp.sum())
        
        prices_old = self.prices.iloc[self.head - 1].values # old prices
        prices = self.prices.iloc[self.head].values # new prices
        tc = self.args.transaction_cost

        # Old holdings
        self.balance = self.total_asset * action[0] # R, first index: cash account. no batch, batches are later sampled from buffer...
        self.holdings = (self.total_asset) * action[1:] // prices_old # R x R^30 // R^30

        # Add fractional differences to balance
        exact_holdings = (self.total_asset) * action[1:] / prices_old
        fract_diff = (exact_holdings - self.holdings)
        assert (fract_diff > 0).all()
        self.balance += np.sum(fract_diff * prices_old)
 
        # Subtract transaction costs
        turnover = (action[1:] - self.weights.squeeze()[1:]) * self.total_asset
        total_tc = np.sum(np.abs(turnover)) * tc
        self.balance -= total_tc

        # calculate asset gains with new prices
        total_asset = self.balance + (prices * self.holdings).sum()
        reward = (total_asset - self.total_asset) / (self.total_asset + 1e-8)
        # reward *= self._reward_scale
        self.total_asset = total_asset
        # self.total_reward = self.args.gamma * self.total_reward + reward #TODO

        # check if at terminal state
        if self.head == len(self.prices) - 1:
            # reward = self.total_reward #TODO
            profit = self.total_asset / self.args.initial_balance - 1.0
            state = self.reset()
            return state, reward, True, {'profit': profit}


        # Update weights (changed because of return dynamics)
        self.weights = np.concatenate(
            [np.array([self.balance]), prices * self.holdings]
        ) / self.total_asset
        self.weights = self.weights.reshape(-1, 1)
        
        # create state vector
        lag_returns = self.prices.iloc[self.head - 1: self.head + 1].pct_change().dropna(axis=0).T.values # shape: [#stocks, #lagged returns]
        self.lag_returns[1:, :] = np.concatenate([self.lag_returns[1:, :-1], lag_returns], axis=1)
        state = np.concatenate([self.lag_returns, self.weights], axis=1)
        return state, reward, False, {}

