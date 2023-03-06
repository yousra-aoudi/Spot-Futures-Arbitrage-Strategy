"""
The below code shows how to use the Backtester class to simulate the trading of the arbitrage strategy using historical
data and evaluate its profitability and risk. The backtest_arbitrage method simulates the trading of the arbitrage
strategy using historical data by iterating over the historical data and making predictions using the model. It then
evaluates the profitability and risk of the arbitrage strategy using the calculated spot returns and futures returns.
"""

from spot_portfolio_construction import SpotPortfolioConstruction
from arbitrage_prediction_using_ml import ArbitragePredictionUsingML
from strategy_replication import underlying_index_df, spot_price_df, futures_price_df
import numpy as np
from deap import creator, base, tools, algorithms


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
            X = prepare_data_for_lstm_model(self.spot_price_df[:i], self.futures_price_df[:i], self.window_size,
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


if '__name__'=='__main__':

    spot_portfolio_construction = SpotPortfolioConstruction(underlying_index_df)
    replication_weights = spot_portfolio_construction.replication_method('full')

    arbitrage_prediction_using_ml = ArbitragePredictionUsingML(spot_price_df, futures_price_df)
    lstm_model = arbitrage_prediction_using_ml.train_lstm_model()

    backtester = Backtester(spot_price_df, futures_price_df, window_size, num_features)
    backtest_results = backtester.backtest_arbitrage(replication_weights, lstm_model)
    print(backtest_results)

"""
The above code shows how to use the Backtester class to simulate the trading of the arbitrage strategy using historical 
data and evaluate its profitability and risk using an LSTM model. The backtest_results variable represents the evaluated
profitability and risk of the arbitrage strategy, including the Sharpe ratio and the maximum drawdown.
"""

"""
Once you have backtested the arbitrage strategy and evaluated its performance, you may want to optimize the parameters 
of the strategy to improve its performance.

One way to optimize the arbitrage strategy is to use a grid search to search for the best set of parameters that 
maximize the Sharpe ratio or other performance metrics. Here's an example of how to perform a grid search:
"""

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
