"""
@author Luis Molina,  May 11 2020
project:  Implied Volatility for NYMEX Nat Gas
program: implied_vol.py
purpose:
1. Scrape option prices from CME website.
2. Download prices and save to python pickle file. Downloaded options expiration calendar
3. Attach underlying prices to the options prices by future maturity contract, attach expiration dates
4. Calculate implied volatilities for each option in file,  assuming a small interest rate for discouting
5. Filter option prices based on cut-off of moneyness
4. In the interest of time I decided to build the surface using calls only or puts only
5. We save files to pickle files,  the code checks to see if the source file exists, if it does not it goes to the CME ftp site
   and downloads the csv file  (had a lot of issues with scraping the actual site,  code is included to show how I would
   have approached the actual scraping,  CME seems to have some restrictions around scraping their quote pages, some of the
   python libraries were unable to get valid response (error code http 404)
6. Utilized an OLS (Ordinary Least squares model provided by statsmodels python library to fit the vol surface to regression line
7. Program prompts user for contract to evaluate:  e.g. 'NGN20'
8.  User can provide any contract maturity to generate vol curve.

Improvements if I had more time:
1.  Add utilize a database server to create Tables (schema) to hold data
2.  Use both Puts and Calls to generate surface, possibly a polynomial based model, risk-reversals, straddles, strangles
3.  Create a surface using moneyness and maturity to show 3D structure and termstructure of volatility skew
4.  More error checking,  unit testing
5.  etc.

"""



import requests
from requests.exceptions import HTTPError
import pandas as pd
import numpy as np
import scipy.stats as ss
import json
import time
import urllib
from splinter import Browser

import logging
from bs4 import BeautifulSoup
from dateutil.parser import parse
import xlwings
from definitions import ROOT_DIR
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.sandbox.regression.predstd import wls_prediction_std

import seaborn
seaborn.set()

import warnings
warnings.simplefilter('ignore')

# debugging information
logging.basicConfig(level=logging.INFO)

class GetData:
    def __init__(self, tradeDate):

        self.futures_sym = 'NG'
        self.options_sym = 'LN'

        self.url = "https://www.cmegroup.com/trading/energy/natural-gas/natural-gas_quotes_settlements_options.html?optionProductId=1352&optionExpiration=1352-U9#optionProductId=1352&optionExpiration=1352-M0&tradeDate=05%2F11%2F2020"
        self.futures_url = "https://www.cmegroup.com/trading/energy/natural-gas/natural-gas_quotes_settlements_futures.html?optionProductId=1352&optionExpiration=1352-K0"
        self.executable_path = {'executable_path': r"chromedriver"}  # to use in situations where the selenium library is used to scrape the CME web page
        self.browser = None
        self.response = None
        self.cme_url = 'ftp://ftp.cmegroup.com/settle'
        self.nymex_file = 'nymex_future.csv'
        self.nymex_options = 'nymex_option.csv'
        self.tradeDate = tradeDate
        self.SRC_FILENAME = '{0}_options_{1}.pkl'.format(self.futures_sym.lower(), self.tradeDate)
        self.SRC_VOL_FILE = '{0}_vol_surface_{1}.pkl'.format(self.futures_sym.lower() , self.tradeDate)
        self.expiry_cal = 'expiry_cal_ng.xlsx'




    def open_browser(self):

        self.browser = Browser('chrome', self.executable_path, headless=False)

    def build_url(self):
        """url build for scraping web page."""

        method = 'GET'
        p = requests.Request(method, self.url).prepare()
        myurl = p.url

        self.open_browser()
        self.browser.visit(myurl)

        logging.info(self.browser.find_by_text("table").copy())
        new_url = self.browser.url

        logging.info(new_url)

        self.browser.quit()

    def get_url(self):
        try:
            self.response = requests.get(self.url)

            self.response.raise_for_status()

        except HTTPError as http_err:
            logging.info(f'HTTP error occurred: {http_err}')

        except Exception as err:
            logging.info(f'Some type of error occurred: {err}')
        else:
            logging.info('Success!')

    def parse_cme_url(self):
        self.get_url()
        soup = BeautifulSoup(self.response.text, 'lxml')
        table_list = [(table['id'], self.parse_html_table(table)) for table in soup.find_all('table')]

        return table_list

    def parse_html_table(self, table):
        """Parse web page table."""
        cols = 0
        rows = 0
        column_names = []

        for row in table.find_all('tr'):
            td_tags = row.find_all('td')
            if len(td_tags) > 0:
                rows += 1
                if cols == 0:
                    cols = len(td_tags)

            # column names
            th_tags = row.find_all('th')
            if len(th_tags) > 0 and len(column_names) == 0:
                for th in th_tags:
                    column_names.append(th.get_text())

            if len(column_names) > 0 and len(column_names) != cols:
                logging.info(len(column_names), cols)
                raise Exception("Columns titles do not match the number of columns")

            columns = column_names if len(column_names) > 0 else range(cols)
            df = pd.DataFrame(columns = columns, index=range(rows))

            row_marker = 0
            for row in table.find_all('tr'):
                column_marker = 0
                columns = row.find_all('td')
                for column in columns:
                    df.iat[row_marker, column_marker] = column.get_text()
                    column_marker += 1
                if len(columns) > 0:
                    row_marker += 1

            # Convert to float if possible
            for col in df:
                try:
                    df[col] = df[col].astype(float)
                except ValueError:
                    pass
            return df

# actual method used to get settle prices from the CME
    def get_cme_settles(self):
        """Parse CSV file from CME ftp website."""

        nymex_futures = pd.read_csv('{0}/{1}'.format(self.cme_url, self.nymex_file))
        nymex_options = pd.read_csv('{0}/{1}'.format(self.cme_url, self.nymex_options))

        d = [(parse(s)).strftime('%Y-%m-%d') for s in nymex_futures['TRADEDATE']]
        o = [(parse(s)).strftime('%Y-%m-%d') for s in nymex_options['TRADEDATE']]

        nymex_futures['TRADEDATE'] = d
        nymex_options['TRADEDATE'] = o

        # tradeDate = nymex_futures['TRADEDATE'].iloc[0]
        # optTradeDate = nymex_options['TRADEDATE'].iloc[0]

        # filter for futures and options
        ng_fut = nymex_futures[nymex_futures['PRODUCT SYMBOL'] == self.futures_sym]
        ng_opt = nymex_options[nymex_options['PRODUCT SYMBOL']== self.options_sym]

        ng_fut.index = [s for s in ng_fut['CONTRACT']]

        ng_opt['CONTRACT'] = [contract.split(" ")[0].replace(self.options_sym, self.futures_sym) for contract in ng_opt['CONTRACT'] ]
        ng_opt['UNDERLYING'] = [ng_fut.loc[row, 'SETTLE'] for row in ng_fut['CONTRACT'] for contract in ng_opt['CONTRACT'] if row==contract]
        expiry_file = pd.read_excel(self.expiry_cal, header=0, index_col=0)

        ng_opt['EXPIRY'] = [expiry_file.loc[row, 'Settlement'] for row in expiry_file.index for contract in ng_opt['CONTRACT'] if row==contract]
        ng_opt['EXPIRY'] = pd.to_datetime(ng_opt['EXPIRY'])

        tradeDate = pd.to_datetime(ng_opt['TRADEDATE'].iloc[0])

        ng_opt['DTE'] = (ng_opt["EXPIRY"] - tradeDate).dt.days
        ng_opt['PUT/CALL'] = np.where(ng_opt['PUT/CALL'].loc[:] == 'C', 'Call', 'Put' )
        ng_opt = ng_opt[ng_opt['SETTLE'] >= 0.005]

        idx = range(len(ng_opt))
        ng_opt.index = idx

        ng_opt = ng_opt[['CONTRACT', 'TRADEDATE', 'UNDERLYING', 'PUT/CALL', 'UNDERLYING', 'STRIKE', 'EXPIRY', 'DTE',   'OPEN', 'HIGH', 'LOW', 'LAST', 'PT CHG', 'SETTLE']]

        ng_opt.to_pickle("{0}/{1}".format(ROOT_DIR, self.SRC_FILENAME))

        return ng_opt


# Options Class inherets from GetData base class
class Options(GetData):
    def __init__(self, tradeDate):
        super(Options, self).__init__(tradeDate)
        self.option_model = 'Black76'
        self.tradeDate = tradeDate

    def GBlackScholes(self, CPflag, S, X, T, r, b, V):
        '''Generalized Black76 European options model for Futures.'''
        Gd1 = (np.log(S / X) + (b + V ** 2 / 2) * T) / (V * np.sqrt(T))
        Gd2 = Gd1 - V * np.sqrt(T)

        if CPflag == "Call":
            return S * np.exp((b - r) * T) * ss.norm.cdf(Gd1) - X * np.exp(-r * T) * ss.norm.cdf(Gd2)
        else:
            return X * np.exp(-r * T) * ss.norm.cdf(-Gd2) - S * np.exp((b - r) * T) * ss.norm.cdf(-Gd1)

    def GDelta(self, CPflag, S, X, T, r, b, V):
        '''Generalized Black_Scholes delta for Options on futures.'''

        Gd1 = (np.log(S / X) + (b + V ** 2 / 2) * T) / (V * np.sqrt(T))

        if CPflag == "Call":
            GDelta = np.exp((b - r) * T) * ss.norm.cdf(Gd1)
        else:
            GDelta = -np.exp((b - r) * T) * ss.norm.cdf(-Gd1)

        return GDelta

    def GVega(self, S, X, T, r, b, V):

        Vd1 = (np.log(S / X) + (b + V ** 2 / 2) * T) / (V * np.sqrt(T))

        return (S * np.exp((b - r) * T) * ss.norm.pdf(Vd1) * np.sqrt(T))

    def GImpliedVolatility(self, CPflag, S, X, T, r, b, cm, epsilon):
        '''Calculate implied volatility using Newton-Raphson method with initial vol seed.'''

        vi = np.sqrt(abs(np.log(S / X) + r * T) * 2 / T)  # initial vol

        ci = self.GBlackScholes(CPflag, S, X, T, r, b, vi)
        vegai = self.GVega(S, X, T, r, b, vi)
        min_Diff = abs(cm - ci)

        while (abs(cm - ci) >= epsilon) & (abs(cm - ci) <= min_Diff):
            vi = vi - (ci - cm) / vegai
            ci = self.GBlackScholes(CPflag, S, X, T, r, b, vi)
            vegai = self.GVega(S, X, T, r, b, vi)
            min_Diff = abs(cm - ci)
        if abs(cm - ci) < epsilon:
            vi = vi
        else:
            vi = 'NA'

        return vi

    def get_prices(self):
        """Get prices from either a stored pickle file or CME ftp site."""

        # get stored prices from Pickle file or Database
        try:
            option_prices = pd.read_pickle("{0}/{1}".format(ROOT_DIR, self.SRC_FILENAME))
        except FileNotFoundError:
            option_prices = self.get_cme_settles()
            option_prices.to_pickle("{0}/{1}".format(ROOT_DIR, self.SRC_FILENAME))

        return option_prices

    def test_func(self, x, a, b):
        """Test function."""
        return a * x

    def plot_vol_surface(self, x, y, contract, put_call):
        """Plot surface using Ordinary Least Square Fit."""

        df = pd.DataFrame(columns=['x', 'y'])
        df['x'] = x
        df['y'] = y
        degree = 3

        try:
            weights = np.polyfit(x, y, degree)
            model = np.poly1d(weights)
            results = smf.ols(formula='y ~ model(x)', data=df).fit()
            prstd, iv_l, iv_u = wls_prediction_std(results)
            fig, ax = plt.subplots(figsize=(8, 6))
            plt.title("Implied Vol for NG European Options = {0}, moneyness plot= log(K/F): tradeDate: {1}".format(contract, self.tradeDate))
            ax.plot(x, y, 'o', label="{0} Implied Vol".format(put_call))
            ax.plot(x, results.fittedvalues, 'r--.', label="OLS")
            ax.plot(x, iv_u, 'r--')
            ax.plot(x, iv_l, 'r--')
            ax.legend(loc='best')
            plt.xlabel("Moneyness: log(K/F)")
            plt.ylabel("Implied Vol.")
            plt.axvline(0, color='k')
            plt.show()

        except ValueError:
            logging.info("ValueError!")

    def plot_raw_vol(self, x, y, contract, put_cal):

        df = pd.DataFrame(columns=['x', 'y'])
        df['x'] = x
        df['y'] = y

        plt.figure()
        plt.scatter(x, y)
        plt.show()

    def vol_surface(self, put_call='Call'):
        """Generate vol surface using Moneyness: log(Strike/Underlying)"""

        contract = input("Enter Natural Gas Contract (e.g. {0}Q20): ".format(self.futures_sym)).upper()
        option_prices = self.get_prices()
        option_prices = option_prices[option_prices['CONTRACT']==contract]

        # filter for moneyness
        option_prices['moneyness'] = np.log(option_prices['STRIKE'] / option_prices['UNDERLYING'])
        option_prices = option_prices[option_prices['moneyness'] <= 0.5]
        df = option_prices[option_prices['moneyness'] >= -0.5]

        rate = 0.007  # low interest rate constant for now,  not material in this low rate environment
        epsilon = 0.001
        b = 0 # for futures

        n = len(df)
        vols = []
        delta = []
        # iterate through each option type and price to generate Implied Vol
        for i in range(n):
            CPflag = df['PUT/CALL'].iloc[i]; F = df['UNDERLYING'].iloc[i]; K = df['STRIKE'].iloc[i]
            T = float(df['DTE'].iloc[i])/365.0; r = rate; cm = df['SETTLE'].iloc[i]
            b= 0   # for futures
            iv = self.GImpliedVolatility(CPflag, F, K, T, r, b, cm, epsilon)
            dlt = self.GDelta(CPflag, F, K, T, r, b, iv)
            vols.append(iv)
            delta.append(dlt)

        df['implied_vol'] = vols
        df['delta'] = delta

        df = df[df['PUT/CALL'] == put_call]
        x = df['moneyness']
        y = df['implied_vol']

        # xlwings.view(df)
        # df = df[df['CONTRACT','DTE','STRIKE', 'moneyness', 'implied_vol']]
        df.to_pickle("{0}/{1}".format(ROOT_DIR, self.SRC_VOL_FILE))

        try:
            result = self.plot_vol_surface(x,y, contract, put_call)
        except  ValueError:
            self.plot_raw_vol(x, y, contract, put_call)

        return df

    def read_vol_surface(self):
        """Read vol surface from pickle file."""

        vol_surface = pd.read_pickle("{0}/{1}".format(ROOT_DIR, self.SRC_VOL_FILE))

        print(vol_surface)

        # xlwings.view(vol_surface)


if __name__ == '__main__':

    tradeDate = '2020-05-12'
    d = Options(tradeDate)
    d.options_sym = 'LN'
    d.futures_sym = 'NG'

    #table = d.parse_cme_url()[0][1]
    #table.head()

    # opt = d.get_cme_settles()
    opt1 = d.vol_surface(put_call='Put')
    xlwings.view(opt1)
    
    d.read_vol_surface()



