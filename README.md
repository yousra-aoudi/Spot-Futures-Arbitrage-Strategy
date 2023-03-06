# Spot-Futures-Arbitrage-Strategy
This repository contains the code for a Spot-Futures Arbitrage Strategy in Python, which aims to provide a flexible and customizable tool for analyzing the pricing relationship between spot prices and futures prices, constructing portfolios to track the trend of futures, analyzing non-arbitrage intervals and profitable arbitrage opportunities, and predicting spread between assets using machine learning models.

Classes

The strategy is based on four main classes:

- SpotFuturesRelationship: Contains methods to investigate the lead-lag relationship between futures and spot prices, analyze the pricing efficiency of the market, and apply econometric models to study the relationship between the two.
- SpotPortfolioConstruction: Includes methods to construct a spot portfolio to track the trend of futures, obtain higher returns through the portfolio with higher fitting accuracy, minimum tracking error, convenient transaction, and lower cost, and replicate the portfolio of constituent stocks or construct the ETF.
- FuturesPricingArbitrageIntervals: Has methods to analyze non-arbitrage intervals and determine profitable arbitrage opportunities based on the deviation between stock index futures and spot stock. The class employs general equilibrium models to develop a closed-end equilibrium pricing model for stock index futures and calculate the upper and lower boundaries of the non-arbitrage intervals through the combination of bid and ask spreads.
- ArbitragePredictionUsingML: Contains methods to predict the spread between assets using machine learning models such as Back Propagation (BP) neural network, Long Short-Term Memory (LSTM) neural network, or support vector regression, and establish an arbitrage strategy.
Usage

To use this strategy, you can simply clone the repository to your local machine and run the main.py file. The file contains examples of how to use each of the classes in the strategy.

Dependencies

The code is written in Python and uses several third-party libraries, including:

NumPy
Pandas
Scikit-learn
TensorFlow
StatsModels
Matplotlib
Seaborn
All required dependencies can be installed using pip and the included requirements.txt file.

Contribution

Feel free to contribute to this project by submitting a pull request. If you have any questions or suggestions, please open an issue on this repository.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Note

This project is currently private. If you would like to have access to it, please send a request via message.
