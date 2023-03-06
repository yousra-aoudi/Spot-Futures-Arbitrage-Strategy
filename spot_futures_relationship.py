"""
The code shows how to use the SpotFuturesRelationship class to test the lead-lag relationship between the S&P500
index spot and futures using the OLS regression model from the statsmodels package. The identify_lead_lag method
identifies the strength of the linkage between the futures price and the spot price based on the correlation coefficient.
"""


class SpotFuturesRelationship:
    def __init__(self, spot_price_df, futures_price_df):
        self.spot_price_df = spot_price_df
        self.futures_price_df = futures_price_df

    def test_relationship(self):
        # use minute-level high-frequency data to test the relationship between intraday price changes of the S&P500
        # index spot and futures
        model = sm.OLS(self.spot_price_df['price'].diff().iloc[1:],
                       sm.add_constant(self.futures_price_df['price'].diff().iloc[1:]))
        results = model.fit()
        return results.summary()

    def identify_lead_lag(self):
        # identify lead-lag relationship between futures and spot prices
        corr_matrix = np.corrcoef(self.spot_price_df['price'], self.futures_price_df['price'])
        corr_coef = corr_matrix[0, 1]
        if corr_coef > 0.5:
            return "There is a strong linkage between the futures price and the spot price"
        else:
            return "There is no strong linkage between the futures price and the spot price"



