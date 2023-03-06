"""
Once you have backtested the arbitrage strategy and evaluated its performance, you may want to optimize the parameters
of the strategy to improve its performance.

One way to optimize the arbitrage strategy is to use a grid search to search for the best set of parameters that
maximize the Sharpe ratio or other performance metrics. Here's an example of how to perform a grid search:
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

spot_portfolio_construction = SpotPortfolioConstruction(underlying_index_df)
replication_weights = spot_portfolio_construction.replication_method('full')

arbitrage_prediction_using_ml = ArbitragePredictionUsingML(spot_price_df, futures_price_df)

parameters = {'window_size': [30, 60, 90], 'num_features': [5, 10, 20]}
results = {}
for window_size in parameters['window_size']:
    for num_features in parameters['num_features']:
        backtester = Backtester(spot_price_df, futures_price_df, window_size, num_features)
        lstm_model = arbitrage_prediction_using_ml.train_lstm_model(window_size, num_features)
        backtest_results = backtester.backtest_arbitrage(replication_weights, lstm_model)
        results[(window_size, num_features)] = backtest_results

best_params = max(results, key=results.get)
best_results = results[best_params]
print(f'Best parameters: window_size={best_params[0]}, num_features={best_params[1]}')
print(f'Best results: {best_results}')

"""
The above code shows how to use a grid search to search for the best set of parameters that maximize the Sharpe ratio 
of the arbitrage strategy. The parameters variable represents the grid of parameters to search over. The results 
dictionary stores the evaluated performance metrics for each set of parameters. The best_params variable represents the
set of parameters that maximizes the Sharpe ratio, and the best_results variable represents the evaluated performance 
metrics for the best set of parameters.

Another way to optimize the arbitrage strategy is to use a machine learning algorithm such as a genetic algorithm or a 
neural network to search for the best set of parameters. Here's an example of how to use a genetic algorithm to optimize
the arbitrage strategy:
"""

from deap import creator, base, tools, algorithms

spot_portfolio_construction = SpotPortfolioConstruction(underlying_index_df)
replication_weights = spot_portfolio_construction.replication_method('full')

arbitrage_prediction_using_ml = ArbitragePredictionUsingML(spot_price_df, futures_price_df)


def evaluate_fitness(params):
    window_size, num_features = params
    backtester = Backtester(spot_price_df, futures_price_df, window_size, num_features)
    lstm_model = arbitrage_prediction_using_ml.train_lstm_model(window_size, num_features)
    backtest_results = backtester.backtest_arbitrage(replication_weights, lstm_model)
    sharpe_ratio = backtest_results['sharpe_ratio']
    return sharpe_ratio,


creator.create('FitnessMax', base.Fitness, weights=(1.0,))
creator.create('Individual', list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register('attr_window_size', random.choice, [30, 60, 90])
toolbox.register('attr_num_features', random.choice, [5, 10, 20])
toolbox.register('individual', tools.initCycle, creator.Individual,
                 (toolbox.attr_window_size, toolbox.attr_num_features), n=1)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
toolbox.register('evaluate', evaluate_fitness)
toolbox.register('mate', tools.cxOnePoint)
toolbox.register('mutate', tools.mutUniformInt, low=30, up=90, indpb=0.5)
toolbox.register('select', tools.selTournament, tournsize=3)

population = toolbox.population(n=10)
NGEN = 10
for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))

best_ind = tools.selBest(population, k=1)[0]
print(f'Best individual: {best_ind}, fitness: {best_ind.fitness.values}')


"""
The above code shows how to use a genetic algorithm to optimize the arbitrage strategy. The evaluate_fitness function
evaluates the fitness of an individual (set of parameters) by performing a backtest using the Backtester and 
ArbitragePredictionUsingML classes. The creator module is used to create the FitnessMax and Individual classes, which 
represent the fitness and individuals of the genetic algorithm, respectively. The toolbox module is used to register the
functions and parameters of the genetic algorithm. The population variable represents the initial population of 
individuals to start the genetic algorithm. The NGEN variable represents the number of generations to run the genetic 
algorithm for. The for loop runs the genetic algorithm for the specified number of generations, and the best_ind 
variable represents the best individual found by the genetic algorithm.

Overall, the process of optimizing the arbitrage strategy involves selecting appropriate performance metrics, defining 
a set of parameters to optimize, and using a search algorithm to find the best set of parameters. By optimizing the 
arbitrage strategy, you can improve its performance and increase your potential profits.
"""

"""
Below you can use the classes and functions described earlier to backtest a spot-futures arbitrage strategy on 
historical data.

First, you need to prepare the data. Let's assume that you have historical spot prices and futures prices for a stock 
index, and you want to backtest a spot-futures arbitrage strategy over a period of one year. Here's how you can load 
the data into a pandas DataFrame:
"""

import pandas as pd

# Load historical spot prices and futures prices into pandas DataFrames
spot_price_df = pd.read_csv('spot_price.csv', index_col='Date', parse_dates=True)
futures_price_df = pd.read_csv('futures_price.csv', index_col='Date', parse_dates=True)
