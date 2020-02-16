"""This script scrapes the tweets for companies and store them in csv files."""

import os

import pandas as pd
from data.config import raw_data_dir, tweets_data_dir

company_twit_info = pd.read_csv(raw_data_dir + '/' + 'company_2014_twitter_info.csv')

for i, row in company_twit_info.iterrows():
    twitter_username = row['twitter_username']
    twit_since = row['twit_since']
    twit_until = row['twit_until']

    if not os.path.exists(tweets_data_dir + '/{}.csv'.format(twitter_username)):
        try:
            os.system('twitterscraper "(from:{0}) until:{1} since:{2}" -o {3}/{0}.csv -c' \
                      .format(twitter_username, twit_until, twit_since, tweets_data_dir))
        except:
            with open(tweets_data_dir + '/tweets/error_companies.txt', 'a') as f:
                f.write('{}, {}, {}\n'.format(twitter_username, twit_since, twit_until))

    # exit()
