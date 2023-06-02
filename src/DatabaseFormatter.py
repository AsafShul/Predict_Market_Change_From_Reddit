import os
import json
import time

import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime as dt

BASE_DATA_DIR = '../data'
STOCKS_DIR = os.path.join(BASE_DATA_DIR, 'reddit', 'stocks')
BETS_DIR = os.path.join(BASE_DATA_DIR, 'reddit', 'wallStreetBets')
WORLD_DIR = os.path.join(BASE_DATA_DIR, 'reddit', 'world_news')

STOCKS_POSTS_PATH = os.path.join(STOCKS_DIR, 'stocks_submissions')
STOCKS_COMMENTS_PATH = os.path.join(STOCKS_DIR, 'stocks_comments')
STOCKS_OUTPUT_PATH = os.path.join(STOCKS_DIR, 'stocks.csv')
BETS_POSTS_PATH = os.path.join(BETS_DIR, 'wallstreetBets_submissions')
BETS_COMMENTS_PATH = os.path.join(BETS_DIR, 'wallstreetBets_comments')
BETS_OUTPUT_PATH = os.path.join(BETS_DIR, 'wallstreetBets.csv')

# STOCKS_POSTS_PATH = r"C:\temp\anlp_project\stocks_submissions"
# STOCKS_COMMENTS_PATH = r"C:\temp\anlp_project\stocks_comments"
# BETS_POSTS_PATH = r"C:\temp\anlp_project\wallstreetBets_submissions"
# BETS_COMMENTS_PATH = r"C:\temp\anlp_project\wallstreetBets_comments"
# STOCKS_OUTPUT_PATH = r"C:\temp\anlp_project\stocks.csv"
# BETS_OUTPUT_PATH = r"C:\temp\anlp_project\bets.csv"

REQUIRED_COLS = ['created_utc', 'selftext', 'name']
SUBMISSION_COLS = ['created_utc', 'id', 'num_comments', 'title', 'selftext', 'score']


class DatabaseFormatter:
    @staticmethod
    # def format(stock_data_path_, bets_data_path_, world_data_path_, labels_path_, output_path_=None):
    def format(stock_data_path_, bets_data_path_, labels_path_, output_path_=None):
        bets_df = pd.read_csv(bets_data_path_, index_col='post_time', parse_dates=True)
        stocks_df = pd.read_csv(stock_data_path_, index_col='post_time', parse_dates=True)
        # world_df = pd.read_csv(world_data_path_, index_col='post_time', parse_dates=True)

        labels = DatabaseFormatter._label_data(labels_path_)

        # df_ = pd.concat([bets_df, stocks_df, world_df], axis=0).sort_index()
        df_ = pd.concat([bets_df, stocks_df], axis=0).sort_index()
        df_.selftext = df_.selftext.fillna(' ')
        df_.title = df_.title.fillna(' ')

        df_['post'] = df_.title + ' $$$ ' + df_.selftext

        df_ = df_[(df_.num_comments < 1000) & (df_.num_comments > 5)]
        df_ = df_[~df_.selftext.str.contains('www.reddit.com/poll')]  # polls
        df_ = df_[~df_.selftext.str.startswith('&amp')]  # probably a memes

        df_ = df_[[d in labels.index for d in df_.index.date]]  # remove non-trading days
        df_['label'] = labels[df_.index.date].values

        df_.drop(columns=['id', 'num_comments', 'title', 'num_comments', 'selftext', 'score'], inplace=True)

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
        labels = ((percent_change.abs() > eps) * np.sign(percent_change)).astype(int).shift(-1).dropna() + 1
        return labels

    @staticmethod
    def format_raw_submission_data(posts_path, comments_path, output_path):
        time.sleep(1)
        print(f'Formatting raw {os.path.basename(posts_path)}...')

        comments_per_post = DatabaseFormatter.get_comments_per_post(posts_path, comments_path)

        lines = []
        with open(posts_path, 'r', encoding='utf8') as f:
            for line_raw in tqdm(f):
                line = json.loads(line_raw)

                if not all(col in line for col in REQUIRED_COLS):
                    continue

                post_datetime = dt.datetime.fromtimestamp(int(line['created_utc']))
                if post_datetime.date().year < 2018:
                    continue
                line['post_time'] = post_datetime
                line.pop('created_utc')

                if line['selftext'] == '[removed]' or line['selftext'] == '[deleted]':
                    continue

                line['num_comments'] = comments_per_post[line['name']]

                # drop keys that are not in THE_BEST
                for key in list(line.keys()):
                    if key not in SUBMISSION_COLS + ['post_time', 'num_comments']:
                        line.pop(key)

                lines.append(line)

        df = pd.DataFrame(lines).set_index('post_time')
        df.to_csv(output_path)
        print(f'Finished creating {os.path.basename(output_path)}!')

    @staticmethod
    def get_comments_per_post(posts_filename, comments_filename):
        print("Counting comments per post...")

        comments_info = DatabaseFormatter.get_id_to_data_dict(comments_filename, ['parent_id', 'created_utc'])
        posts_info = DatabaseFormatter.get_id_to_data_dict(posts_filename, ['created_utc'])

        parent_to_children_info = {}
        for post_id, (created,) in tqdm(posts_info.items()):
            parent_to_children_info[post_id] = [created, []]

        for comment_id, (parent_id, created) in tqdm(comments_info.items()):
            parent_to_children_info.setdefault(parent_id, [None, []])[1].append(comment_id)
            parent_to_children_info[comment_id] = [created, []]

        results = {}
        for post_id, (created,) in tqdm(posts_info.items()):
            created_day = dt.datetime.fromtimestamp(int(created)).date()
            results[post_id] = DatabaseFormatter._get_comments_per_post(post_id, created_day, parent_to_children_info)

        print("Finished counting comments per post.")

        return results

    @staticmethod
    def _get_comments_per_post(post_id, created_day, parent_to_children_info):
        num_comments = 0
        for comment_id in parent_to_children_info[post_id][1]:
            comment_day = dt.datetime.fromtimestamp(int(parent_to_children_info[comment_id][0])).date()
            if comment_day == created_day:
                num_comments += 1
                num_comments += DatabaseFormatter._get_comments_per_post(comment_id, created_day, parent_to_children_info)
        return num_comments

    @staticmethod
    def get_id_to_data_dict(filename, data_keys):
        id_to_data = {}
        with open(filename, 'r', encoding='utf8') as f:
            for line in tqdm(f):
                # line = ' '.join(line.replace('"', "").split())
                info = json.loads(line)
                if 'name' not in info or not all([key in info.keys() for key in data_keys]):
                    continue
                id_to_data[info['name']] = [info[key] for key in data_keys]

        return id_to_data


if __name__ == '__main__':

    # format:
    DatabaseFormatter.format_raw_submission_data(STOCKS_POSTS_PATH, STOCKS_COMMENTS_PATH, STOCKS_OUTPUT_PATH)
    # DatabaseFormatter.format_raw_submission_data(BETS_POSTS_PATH, BETS_COMMENTS_PATH, BETS_OUTPUT_PATH)

    # DatabaseFormatter.format_raw_submission_data(os.path.join(STOCKS_DIR, 'stocks_submissions'),
    #                                              os.path.join(STOCKS_DIR, 'stocks_comments'),
    #                                              os.path.join(STOCKS_DIR, f'stocks_submissions.csv'))
    # DatabaseFormatter.format_raw_submission_data(BETS_DIR, 'wallstreetbets_submissions')
    # DatabaseFormatter.format_raw_submission_data(WORLD_DIR, 'worldnews_submissions')

    # bets_path = '../data/reddit/wallStreetBets/wallstreetbets_submissions.csv'
    # stocks_path = '../data/reddit/stocks/stocks_submissions.csv'
    # # world_path = '../data/reddit/world_news/worldnews_submissions.csv'
    #
    # labels_path = '../data/spy/spy.csv'
    #
    # output_path = '../data/formatted_df.ftr.zstd'
    #
    # df = DatabaseFormatter.format(stock_data_path_=stocks_path, bets_data_path_=bets_path,
    #                               # world_data_path_=world_path, labels_path_=labels_path, output_path_=output_path)
    #                               labels_path_=labels_path, output_path_=output_path)

    print('Done')
