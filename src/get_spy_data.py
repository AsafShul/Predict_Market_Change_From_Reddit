#!%PYTHON_HOME%\python.exe
# coding: utf-8

from datetime import datetime
import pandas as pd
import yfinance


INTERESTING_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']
DATA_BEGINNING = datetime(2010, 1, 1)
DATA_END = datetime(2023, 6, 1)
SPY_TICKER = 'SPY'


def get_ticker_data(ticker, start=DATA_BEGINNING, end=DATA_END, verbose=False):
    df = yfinance.download(ticker,
                           period=None, start=start, end=end,
                           group_by='ticker', actions=False,
                           auto_adjust=True, back_adjust=False,
                           progress=verbose)

    df = df[INTERESTING_COLUMNS]
    df.columns = df.columns.str.lower()
    # df.index = df.index.strftime(DATETIME_FORMAT)  # Leave it as datetimes, since I changed the table dates
    df.index.name = 'date'
    df = df[~pd.isnull(df.close)]
    return df


def main():
    df = get_ticker_data(SPY_TICKER)
    df.to_csv('spy.csv')
    print("Data exported to spy.csv")


if __name__ == '__main__':
    main()
