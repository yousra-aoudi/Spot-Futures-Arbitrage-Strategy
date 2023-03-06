"""
The code shows how to use the FuturesPricingArbitrageIntervals class to calculate the non-arbitrage interval
based on the carrying cost pricing model. The calculate_arbitrage_interval method calculates the standard deviation and
mean of the carry cost and sets the upper and lower bounds based on the two standard deviations from the mean.
"""
import numpy as np


class FuturesPricingArbitrageIntervals:
    def __init__(self, futures_price_df, spot_price_df):
        self.futures_price_df = futures_price_df
        self.spot_price_df = spot_price_df

    def calculate_arbitrage_interval(self):
        # calculate non-arbitrage interval based on carrying cost pricing model
        carry_cost = self.futures_price_df['price'] - self.spot_price_df['price']
        std = np.std(carry_cost)
        mean = np.mean(carry_cost)
        upper_bound = mean + 2 * std
        lower_bound = mean - 2 * std
        return lower_bound, upper_bound

    def general_equilibrium_model(self):
        # implement general equilibrium model to calculate non-arbitrage interval
        pass


