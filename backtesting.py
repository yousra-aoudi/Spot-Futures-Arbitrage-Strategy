"""
This code shows how to use the Backtester class to simulate the trading of the arbitrage strategy using historical
data and evaluate its profitability and risk. The backtest_arbitrage method simulates the trading of the arbitrage
strategy using historical data by iterating over the historical data and making predictions using the model. It then
evaluates the profitability and risk of the arbitrage strategy using the calculated spot returns and futures returns.

Once you have implemented the trading algorithm, you can backtest the arbitrage strategy to evaluate its performance.
Backtesting involves simulating the trading of the arbitrage strategy using historical data to evaluate its
profitability and risk.
"""


import pandas as pd
import numpy as np

# 1 - import YahooFinanceDataLoader class
from yahoo_finance_data_loader import YahooFinanceDataLoader

# 2 - import Spot Futures Relationship
from spot_futures_relationship import SpotFuturesRelationship

# 2 - import Spot Futures Relationship
from spot_portfolio_construction import SpotPortfolioConstruction

# 3 - import Future pricing arbitrage class
from futures_pricing_arbitrage_intervals import FuturesPricingArbitrageIntervals

# 4 - import arbitrage prediction using ML class
from arbitrage_prediction_using_ml import ArbitragePredictionUsingML

# 5 - import trading algorithm
from trading_algorithm import TradingAlgorithm

yahoo_finance = YahooFinanceDataLoader()
spot_price_df = pd.read_csv('spot_price.csv')
futures_price_df = pd.read_csv('futures_price.csv')
underlying_index_df = pd.read_csv('underlying_index.csv')

spot_portfolio_construction = SpotPortfolioConstruction(underlying_index_df)
replication_weights = spot_portfolio_construction.replication_method('full')

arbitrage_prediction_using_ml = ArbitragePredictionUsingML(spot_price_df, futures_price_df)
lstm_model = arbitrage_prediction_using_ml.train_lstm_model()

# make prediction using LSTM model
X_test = lstm_model.prepare_data_for_lstm_model(spot_price_df, futures_price_df, window_size=None, num_features=None)
spread_prediction = lstm_model.predict(X_test)

# execute trades using trading algorithm
trading_algorithm = TradingAlgorithm(spot_price_df, futures_price_df)
trading_algorithm.trade_arbitrage(replication_weights, spread_prediction)


class Backtester:
    def __init__(self, spot_price_df, futures_price_df, window_size, num_features):
        self.spot_price_df = spot_price_df
        self.futures_price_df = futures_price_df
        self.window_size = window_size
        self.num_features = num_features

    def backtest_arbitrage(self, spot_portfolio_weights, model):
        # simulate trading of the arbitrage strategy using historical data
        spot_positions = []
        futures_positions = []
        for i in range(self.window_size, len(self.spot_price_df)):
            X = lstm_model.prepare_data_for_lstm_model(self.spot_price_df[:i], self.futures_price_df[:i], self.window_size,
                                            self.num_features)
            spread_prediction = model.predict(X)[-1]
            spot_positions.append(spot_portfolio_weights * total_portfolio_value)
            futures_positions.append(spread_prediction * spot_positions[-1] / futures_price_df['price'][i])
            spot_positions[-1] = adjust_spot_positions(spot_positions[-1])
            futures_positions[-1] = adjust_futures_positions(futures_positions[-1])
            execute_spot_trades(spot_positions[-1])
            execute_futures_trades(futures_positions[-1])

        # evaluate profitability and risk of the arbitrage strategy
        spot_returns = self.spot_price_df['returns'][self.window_size:].values
        futures_returns = (self.futures_price_df['price'][self.window_size:].values - self.futures_price_df['price'][
                                                                                      self.window_size - 1:-1].values) / \
                          self.futures_price_df['price'][self.window_size - 1:-1].values
        strategy_returns = np.array(spot_returns) - np.array(futures_returns)
        sharpe_ratio = calculate_sharpe_ratio(strategy_returns)
        max_drawdown = calculate_max_drawdown(strategy_returns)
        return {'sharpe_ratio': sharpe_ratio, 'max_drawdown': max_drawdown}


if __name__ == '__main__':

    spot_portfolio_construction = SpotPortfolioConstruction(underlying_index_df)
    replication_weights = spot_portfolio_construction.replication_method('full')

    arbitrage_prediction_using_ml = ArbitragePredictionUsingML(spot_price_df, futures_price_df)
    lstm_model = arbitrage_prediction_using_ml.train_lstm_model()

    backtester = Backtester(spot_price_df, futures_price_df, window_size=None, num_features=None)
    backtest_results = backtester.backtest_arbitrage(replication_weights, lstm_model)
    print(backtest_results)

    """
    The above code shows how to use the Backtester class to simulate the trading of the arbitrage strategy using 
    historical data and evaluate its profitability and risk using an LSTM model. The backtest_results variable 
    represents the evaluated profitability and risk of the arbitrage strategy, including the Sharpe ratio and the 
    maximum drawdown.
    """