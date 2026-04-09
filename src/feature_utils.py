import numpy as np
import pandas as pd
import datetime
import yfinance as yf
import pandas_datareader.data as web
import requests
#from datetime import datetime, timedelta
import os
import sys
import json #

from src.Custom_Classes import FeatureEngineer


def extract_features():

    return_period = 5
    
    START_DATE = (datetime.date.today() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
    END_DATE = datetime.date.today().strftime("%Y-%m-%d")
    stk_tickers = ['MSFT', 'IBM', 'GOOGL']
    ccy_tickers = ['DEXJPUS', 'DEXUSUK']
    idx_tickers = ['SP500', 'DJIA', 'VIXCLS']
    
    stk_data = yf.download(stk_tickers, start=START_DATE, end=END_DATE, auto_adjust=False)
    #stk_data = web.DataReader(stk_tickers, 'yahoo')
    ccy_data = web.DataReader(ccy_tickers, 'fred', start=START_DATE, end=END_DATE)
    idx_data = web.DataReader(idx_tickers, 'fred', start=START_DATE, end=END_DATE)

    Y = np.log(stk_data.loc[:, ('Adj Close', 'MSFT')]).diff(return_period).shift(-return_period)
    Y.name = Y.name[-1]+'_Future'
    
    X1 = np.log(stk_data.loc[:, ('Adj Close', ('GOOGL', 'IBM'))]).diff(return_period)
    X1.columns = X1.columns.droplevel()
    X2 = np.log(ccy_data).diff(return_period)
    X3 = np.log(idx_data).diff(return_period)

    X = pd.concat([X1, X2, X3], axis=1)
    
    dataset = pd.concat([Y, X], axis=1).dropna().iloc[::return_period, :]
    Y = dataset.loc[:, Y.name]
    X = dataset.loc[:, X.columns]
    dataset.index.name = 'Date'
    #dataset.to_csv(r"./test_data.csv")
    features = dataset.sort_index()
    features = features.reset_index(drop=True)
    features = features.iloc[:,1:]
    return features

def extract_features_pair():

    START_DATE = (datetime.date.today() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
    END_DATE = datetime.date.today().strftime("%Y-%m-%d")
    stk_tickers = ['AAPL', 'MPWR']
    
    stk_data = yf.download(stk_tickers, start=START_DATE, end=END_DATE, auto_adjust=False)

    Y = stk_data.loc[:, ('Adj Close', 'AAPL')]
    Y.name = 'AAPL'

    X = stk_data.loc[:, ('Adj Close', 'MPWR')]
    X.name = 'MPWR'

    dataset = pd.concat([Y, X], axis=1).dropna()
    Y = dataset.loc[:, Y.name]
    X = dataset.loc[:, X.name]
    dataset.index.name = 'Date'
    features = dataset.sort_index()
    features = features.reset_index(drop=True)
    return features

def get_bitcoin_historical_prices(days = 60):
    
    BASE_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    
    params = {
        'vs_currency': 'usd',
        'days': days,
        'interval': 'daily' # Ensure we get daily granularity
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['Timestamp', 'Close Price (USD)'])
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms').dt.normalize()
    df = df[['Date', 'Close Price (USD)']].set_index('Date')
    return df

def convert_input_pca_regression(request_body, request_content_type):
    print(f"Receiving data of type: {request_content_type}")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    file_path = os.path.join(project_root, 'Portfolio/SP500Data.csv')

    dataset = pd.read_csv(file_path,index_col=0)
    target = 'IBM'

    
    option = 1
    
    if option == 1:

        
        price = dataset[target].copy()
        
        X = pd.DataFrame(index=dataset.index)
        
        # Returns
        X['ret_1'] = price.pct_change(1)
        X['ret_2'] = price.pct_change(2)
        X['ret_3'] = price.pct_change(3)
        X['ret_5'] = price.pct_change(5)
        X['ret_10'] = price.pct_change(10)
        
        # Lagged returns
        X['ret_1_lag1'] = X['ret_1'].shift(1)
        X['ret_1_lag2'] = X['ret_1'].shift(2)
        X['ret_1_lag3'] = X['ret_1'].shift(3)
        
        # Moving averages
        X['sma_5'] = price.rolling(5).mean()
        X['sma_10'] = price.rolling(10).mean()
        X['sma_20'] = price.rolling(20).mean()
        
        # Ratios
        X['price_sma5_ratio'] = price / X['sma_5']
        X['price_sma10_ratio'] = price / X['sma_10']
        X['price_sma20_ratio'] = price / X['sma_20']
        
        # Volatility
        X['vol_5'] = X['ret_1'].rolling(5).std()
        X['vol_10'] = X['ret_1'].rolling(10).std()
        X['vol_20'] = X['ret_1'].rolling(20).std()
        
        # Rolling highs/lows
        X['roll_max_5'] = price.rolling(5).max()
        X['roll_min_5'] = price.rolling(5).min()
        
        # Distance from extremes
        X['dist_max_5'] = price / X['roll_max_5'] - 1
        X['dist_min_5'] = price / X['roll_min_5'] - 1
        
        # Momentum
        X['mom_5'] = price - price.shift(5)
        X['mom_10'] = price - price.shift(10)
        
        # EMA
        X['ema_5'] = price.ewm(span=5, adjust=False).mean()
        X['ema_10'] = price.ewm(span=10, adjust=False).mean()
        X['ema_ratio'] = X['ema_5'] / X['ema_10']
        
        # RSI-style
        delta = price.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        
        rs = avg_gain / avg_loss
        X['rsi_14'] = 100 - (100 / (1 + rs))
        
        
        technicalindicator_1 = 'mom_5'
        mom_5 = json.loads(request_body)[technicalindicator_1]
    
        technicalindicator_2 = 'ret_5'
        ret_5 = json.loads(request_body)[technicalindicator_2]
            
        # Calculate the distance
        distances = np.sqrt(
        (X[technicalindicator_1] - mom_5)**2 +
        (X[technicalindicator_2] - ret_5)**2
        )
            
            
        closest_index = distances.idxmin()
        closest_row = X.loc[[closest_index]]
        closest_row[technicalindicator_1] = mom_5
        closest_row[technicalindicator_2] = ret_5
        
        return closest_row
    else:
    
        return_period = 5

        SP500_1 = 'IBM_CR_Cum'
        IBM_CR_Cum = json.loads(request_body)[SP500_1]
        SP500_2 = 'NVDA_CR_Cum'
        NVDA_CR_Cum = json.loads(request_body)[SP500_2]

        X = np.log(dataset.drop([target],axis=1)).diff(return_period)
        X = np.exp(X).cumsum()
        X.columns = [name + "_CR_Cum" for name in X.columns]
        
        # Calculate the distance
        distances = np.sqrt(
            (X[SP500_1] - IBM_CR_Cum)**2 + 
            (X[SP500_2] - NVDA_CR_Cum)**2
        )
        
        closest_index = distances.idxmin()
        closest_row = X.loc[[closest_index]]
    
        closest_row[SP500_1] = IBM_CR_Cum
        closest_row[SP500_2] = NVDA_CR_Cum
    
        return closest_row

