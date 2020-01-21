"""This script scrapes the tweets for companies and store them in csv files."""
from data.config import raw_data_dir, processed_data_dir
import pandas as pd
import os

processed_data_dir = '~/Dropbox/TEMP'

company_twit_info = pd.read_csv(processed_data_dir + '/' + 'company_2014_twitter_info.csv')

for i, row in company_twit_info.iterrows():
    twitter_username = row['twitter_username']
    twit_since = row['twit_since']
    twit_until = row['twit_until']

    if not os.path.exists(processed_data_dir + '/tweets/{}.csv'.format(twitter_username)):
        # sample twitterscraper query
        # twitterscraper "(from:Cyclica) until:2018-01-01 since:2017-01-01" -d

        try:
            # os.system('twitterscraper "(from:{0}) until:{1} since:{2}" -d ' \
            # .format(twitter_username, twit_until, twit_since))
            os.system('twitterscraper "(from:{0}) until:{1} since:{2}" -o {3}/tweets/{0}.csv -c' \
                      .format(twitter_username, twit_until, twit_since, processed_data_dir))
        except:
            with open(processed_data_dir + '/tweets/error_companies.txt', 'a') as f:
                f.write('{}, {}, {}\n'.format(twitter_username, twit_since, twit_until))

    # exit()
