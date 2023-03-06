"""
This code shows how to use the SpotPortfolioConstruction class to construct a spot portfolio using the replication
method of constituent stocks or the construction method of ETF. The replication_method method allows the user to choose
between full replication, sampling replication, and hierarchical replication. The etf_construction method can be
implemented to construct an ETF based on the underlying index.
"""
import numpy as np


class SpotPortfolioConstruction:
    def __init__(self, underlying_index_df):
        self.underlying_index_df = underlying_index_df

    def replication_method(self, replication_type='full'):
        # use replication method of constituent stocks to construct spot portfolio
        if replication_type == 'full':
            weights = np.ones(len(self.underlying_index_df)) / len(self.underlying_index_df)
            return weights
        elif replication_type == 'sampling':
            sampled_indices = np.random.choice(len(self.underlying_index_df), int(len(self.underlying_index_df) / 2),
                                               replace=False)
            weights = np.zeros(len(self.underlying_index_df))
            weights[sampled_indices] = 1 / len(sampled_indices)
            return weights
        elif replication_type == 'hierarchical':
            # implement hierarchical replication method
            pass

    def etf_construction(self, etf_name):
        # use construction method of ETF to construct spot portfolio
        pass


