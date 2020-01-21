"""This script scrapes the tweets for companies and store them in csv files."""
import pandas as pd
import os

raw_data_dir = '../data/raw'
processed_data_dir = '../data/processed'

company_twit_info = pd.read_csv(processed_data_dir + '/' + 'company_2014_twitter_info.csv')
# print(company_twit_info)

for i, row in company_twit_info.iterrows():

    twitter_username = row['twitter_username']
    twit_since = row['twit_since']
    twit_until = row['twit_until']

    # sample twitterscraper query:
    # twitterscraper "(from:Cyclica) until:2018-01-01 since:2017-01-01" -o Cyclica.csv -d

    print('twitterscraper "(from:{0}) until:{1} since:{2}" -o {3}/tweets/{0}.csv -c'\
              .format(twitter_username, twit_until, twit_since, processed_data_dir))
    # os.system('twitterscraper "(from:{0}) until:{1} since:{2}" -o {3}/tweets/{0}.csv -c'\
    #           .format(twitter_username, twit_until, twit_since, processed_data_dir))
    exit()