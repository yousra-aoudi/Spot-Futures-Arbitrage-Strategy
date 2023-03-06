"""
The YahooFinanceDataLoader class takes four parameters in its constructor: spot_ticker, futures_ticker, start_date, and
end_date. The interval parameter is optional and defaults to '1d', which specifies daily data.

The load_data method downloads historical spot prices and futures prices for the specified tickers and date range from
Yahoo Finance using the yf.download function. The data is then processed to rename the columns of the DataFrames to
'Spot' and 'Futures', respectively, and to combine the spot price and futures price data into a single DataFrame.
Finally, the combined DataFrame is returned.

The save_to_csv method loads the historical spot prices and futures prices using the load_data method, and then extracts
the spot prices and futures prices into separate DataFrames. These DataFrames are then saved to separate CSV files
using the to_csv method of the pandas DataFrame.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm

import yfinance as yf
import pandas as pd

import os
# Get the current working directory
cwd = os.getcwd()
print(cwd)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class YahooFinanceDataLoader:
    def __init__(self, spot_ticker, futures_ticker, spot_price_df, futures_price_df, price_df, start_date, end_date,
                 interval='1d'):
        self.underlying_index_df = None
        self.constituent_data = None
        self.spot_ticker = spot_ticker
        self.futures_ticker = futures_ticker
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.spot_price_df = spot_price_df
        self.futures_price_df = futures_price_df
        self.price_df = price_df

    def load_data(self):
        # Download historical spot prices and futures prices from Yahoo Finance
        self.spot_price_df = pd.DataFrame(yf.download(self.spot_ticker, start=self.start_date, end=self.end_date,
                                                      interval=self.interval)['Adj Close'])
        print(self.spot_price_df.head())
        self.futures_price_df = pd.DataFrame(yf.download(self.futures_ticker, start=self.start_date, end=self.end_date,
                                                         interval=self.interval)['Adj Close'])
        # Rename the columns of the DataFrames
        self.spot_price_df = self.spot_price_df.rename(columns={'Adj Close': 'Spot'})
        self.futures_price_df = self.futures_price_df.rename(columns={'Adj Close': 'Futures'})

        # Combine the spot price and futures price data into a single DataFrame
        self.price_df = pd.concat([self.spot_price_df, self.futures_price_df], axis=1)

        return self.price_df

    def get_index_data(self):
        # Get the current SP components, and get a tickers list
        sp_assets = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        assets = sp_assets['Symbol'].str.replace('.', '-').tolist()

        # Use the certificate file for SSL verification
        print(os.path.isfile(cafile))
        ssl_context = ssl.create_default_context(cafile="/Users/yousraaoudi/PycharmProjects/dashboard")

        # Download historical data to a multi-index DataFrame
        try:
            data = yf.download(assets, start=self.start_date, end=self.end_date, as_panel=False,
                               ssl_context=ssl_context)
            filename = 'sp_components_data.csv'
            data.to_csv(filename)
            print('Data saved at {}'.format(filename))
        except ValueError:
            print('Failed download, try again.')
            data = None
        return data

    def save_to_csv(self, spot_filename, futures_filename, underlying_index_filename):
        # Load historical spot prices and futures prices into pandas DataFrames
        self.price_df = self.load_data()
        self.spot_price_df = price_df[['Spot']]
        self.futures_price_df = price_df[['Futures']]
        self.constituent_data = self.get_index_data()

        # Load the underlying index data into a DataFrame
        self.underlying_index_df = pd.DataFrame(self.constituent_data)

        # Save the DataFrames to CSV files
        self.spot_price_df.to_csv(spot_filename, index=True, header=True)
        self.futures_price_df.to_csv(futures_filename, index=True, header=True)
        self.underlying_index_df.to_csv(underlying_index_filename, index=True, header=True)


"""
This code creates a YahooFinanceDataLoader object with the specified tickers and date range, and calls the save_to_csv 
method to download, process, and save the historical price data to separate CSV files. Note that the CSV files will be 
saved in the current working directory.
"""
if __name__ == '__main':
    spot_ticker = '^GSPC'  #S&P 500 index
    futures_ticker = 'ES=F'  #E-mini S&P 500 futures
    start_date = '2021-01-01'
    end_date = '2022-02-15'
    underlying_filename = 'underlying_index.csv'
    spot_filename = 'spot_price.csv'
    futures_filename = 'futures_price.csv'

    # Instantiate the YahooFinanceDataLoader class
    data_loader = YahooFinanceDataLoader(spot_ticker, futures_ticker, start_date, end_date)
    # Load data
    price_df = data_loader.load_data()
    # Save data
    data_loader.save_to_csv(spot_filename, futures_filename, underlying_filename)


