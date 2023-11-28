
import pandas as pd
from openbb_terminal.sdk import openbb
from constants import *
from urllib import request

def get_fred_data(macro_tickers, start_date, end_date):
    macro_data = openbb.economy.fred(macro_tickers, start_date=start_date, end_date=end_date)
    return macro_data[0]


def get_yahoo_data(etf_tickers, start_date, end_date):
    etf_data = pd.DataFrame()
    for ticker in etf_tickers:
        ticker_data = openbb.stocks.load(ticker,
                                         start_date=str(start_date),
                                         end_date=str(end_date),
                                         source=YAHOO_DATA_SOURCE)['Adj Close']
        ticker_data.name = ticker
        etf_data = pd.concat([etf_data, ticker_data], axis=1)
        etf_data.dropna(how='any', inplace=True)
    return etf_data

def get_shiller_data(url, sheet_name, skiprows):
    df = pd.read_excel(request.urlretrieve(url)[0], sheet_name=sheet_name, skiprows=skiprows)
    df.drop(index=df.index[:2], axis=0, inplace=True)
    df.drop(index=df.index[-1],axis=0, inplace=True)
    df['Unnamed: 0'] = df['Unnamed: 0'].astype(str).str.replace('\.1$', '.10')
    df['Unnamed: 0'] = df['Unnamed: 0'].astype(str).str.replace('.', '-')
    df.set_index('Unnamed: 0', inplace=True)
    df.index.rename('Date', inplace=True)
    df.index = pd.to_datetime(df.index)
    df['S&P'] = df['S&P'].astype(float)
    return df



if __name__=="__main__":
    # get_fred_data(['INDPRO', 'PAYEMS'], start_date=START_DATE, end_date=END_DATE)
    boo = METALS_RATIO + BOND_RATIO + EQUITY_RATIO
    print(get_yahoo_data(boo, start_date=START_DATE, end_date=END_DATE))
    # get_shiller_data(SHILLER_URL, SHILLER_SHEET_NAME, SHILLER_SKIP_ROW)