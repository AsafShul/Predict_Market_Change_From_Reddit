import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime as dt

BASE_DATA_DIR = '../data'
STOCKS_DIR = os.path.join(BASE_DATA_DIR, 'reddit', 'stocks')
BETS_DIR = os.path.join(BASE_DATA_DIR, 'reddit', 'wallStreetBets')

SUBMISSION_COLS = ['created_utc', 'id', 'num_comments', 'title', 'selftext', 'score']

class DatabaseFormatter:
    @staticmethod
    def format(stock_data_path_, bets_data_path_, labels_path_, output_path_=None):
        bets_df = pd.read_csv(bets_data_path_, index_col='date_index', parse_dates=True)
        stocks_df = pd.read_csv(stock_data_path_, index_col='date_index', parse_dates=True)
        labels = DatabaseFormatter._label_data(labels_path_)

        df_ = pd.concat([bets_df, stocks_df], axis=0)
        df_ = df_[(df_.num_comments < 1000) & (df_.num_comments > 5)]
        df_ = df_[~df_.selftext.str.contains('www.reddit.com/poll')]  # polls
        df_ = df_[~df_.selftext.str.startswith('&amp')]  # probably a memes

        df_ = df_[[d in labels.index for d in df_.index.date]]  # remove non-trading days
        df_['label'] = labels[df_.index.date].values

        df_.drop(columns=['id', 'num_comments'], inplace=True)

        if output_path_ is not None:
            df_.reset_index().to_feather(output_path_, compression='zstd')

        return df_

    @staticmethod
    def _format_reddit_data(path):
        pass

    @staticmethod
    def _label_data(path, eps=0.5):
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date']).dt.date

        df = df.set_index('date')
        percent_change = (1 - df.close / df.open) * 100
        labels = ((percent_change.abs() > eps) * np.sign(percent_change)).astype(int).shift(-1).dropna()
        return labels

    @staticmethod
    def format_raw_submission_data(dir_path, filename):
        print(f'Formatting raw {filename}...')

        lines = []
        with open(os.path.join(dir_path, filename), 'r') as f:
            for line_raw in tqdm(f):
                line = json.loads(line_raw)

                if 'created_utc' in line.keys():
                    post_datetime = dt.datetime.fromtimestamp(int(line['created_utc']))
                    if post_datetime.date().year < 2018:
                        continue
                    line['date_index'] = post_datetime
                    line.pop('created_utc')
                else:
                    continue
                if 'selftext' in line.keys():
                    if line['selftext'] == '[removed]' or line['selftext'] == '[deleted]' or line['selftext'] == '':
                        continue
                else:
                    continue

                # drop keys that are not in THE_BEST
                for key in list(line.keys()):
                    if key not in SUBMISSION_COLS + ['date_index']:
                        line.pop(key)

                lines.append(line)

            df = pd.DataFrame(lines).set_index('date_index')
            df.to_csv(os.path.join(dir_path, f'{filename}.csv'))
            print()


if __name__ == '__main__':

    # format:
    # DatabaseFormatter.format_raw_submission_data(STOCKS_DIR, 'stocks_submissions')
    # DatabaseFormatter.format_raw_submission_data(BETS_DIR, 'wallstreetbets_submissions')

    bets_path = '../data/reddit/wallStreetBets/wallstreetbets_submissions.csv'
    stocks_path = '../data/reddit/stocks/stocks_submissions.csv'
    labels_path = '../data/spy/spy.csv'
    output_path = '../data/formatted_df.ftr.zstd'

    df = DatabaseFormatter.format(stock_data_path_=stocks_path, bets_data_path_=bets_path,
                                  labels_path_=labels_path, output_path_=output_path)

    print()
