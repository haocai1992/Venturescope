"""This script analyze the tweets (before vs. after series A) for one company (that has passed series B)
and generates 'twitter scores' for a company."""
from data.config import raw_data_dir, processed_data_dir, cleaned_data_dir, tweets_data_dir
import pandas as pd
import os
import ast

# cb dataframe for all companies that passed series B and has tweeted between series A and B in the year of 2014.
companies_path = processed_data_dir + '/companies_2014_series_b_tweeted.csv'
companies_series_b_tweeted = pd.read_csv(companies_path)

# functions to generate features/scores for a company's tweeting behavior before and after Series A.
class CompanyTweet:
    """A class to store company's tweeting behavior data."""
    def __init__(self, twitter_username):
        self.series_a_datetime = self.get_series_a_datetime(twitter_username)
        self.tweets = self.get_tweets(twitter_username)
        self.preA_tweets = self.tweets[self.tweets.timestamp < self.series_a_datetime]
        self.postA_tweets = self.tweets[self.tweets.timestamp >= self.series_a_datetime]
        self.preA_timespan = (self.series_a_datetime.date() - self.preA_tweets.timestamp.dt.date.min()).days
        self.postA_timespan = (self.postA_tweets.timestamp.dt.date.max() - self.series_a_datetime.date()).days

    @staticmethod
    def get_series_a_datetime(twitter_username):
        """look up companies df and returns the datetime of series A for this company."""
        series_a_datetime = companies_series_b_tweeted\
                            .loc[companies_series_b_tweeted.twitter_username == twitter_username, 'first_funding_at']\
                            .values[0]
        return pd.to_datetime(series_a_datetime)

    @staticmethod
    def get_tweets(twitter_username):
        """look up the tweets directory and returns the tweet df for this company."""
        if not os.path.exists(tweets_data_dir + '/{}.csv'.format(twitter_username)):
            return None
        else:
            dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
            tweets_df = pd.read_csv(tweets_data_dir + '/{}.csv'.format(twitter_username), delimiter=';',
                                    parse_dates=['timestamp'], date_parser=dateparse)
            return tweets_df

    @staticmethod
    def get_tweet_num(tweets):
        """get the number of tweets."""
        return len(tweets)

    @staticmethod
    def get_tweet_freq(tweets, timespan):
        """get the frequency of tweets (weekly)."""
        tweet_freq_weekly = float(len(tweets)) * 7.0 / float(timespan)
        return tweet_freq_weekly

    @staticmethod
    def get_tweet_avglength(tweets):
        """get the average length of tweets."""
        return tweets.text.apply(len).mean()

    @staticmethod
    def get_tweet_content_richness(tweets):
        """get the average 'tweet content richness'.
        features combined: links, hashtags, has_media, img_urls, video_url."""
        tweet_content_richness = tweets[['links', 'hashtags', 'img_urls']]\
                                 .applymap(lambda x: len(ast.literal_eval(x)))\
                                 .sum(axis=1) + \
                                 tweets.has_media.astype(int) + \
                                 tweets.video_url.fillna(0).astype(int)
        avg_richness = tweet_content_richness.mean()
        return avg_richness

    @staticmethod
    def get_tweet_interactiveness(tweets):
        """get the average 'tweet interactiveness'.
        features combined: likes, retweets, replies, is_replied, is_reply_to, reply_to_users."""
        tweet_interactiveness = tweets[['likes', 'retweets', 'replies']].sum(axis=1) + \
                                tweets[['is_replied', 'is_reply_to']].astype(int).sum(axis=1) + \
                                tweets['reply_to_users'].apply(lambda x: len(ast.literal_eval(x)))
        avg_interactiveness = tweet_interactiveness.mean()
        return avg_interactiveness

    def comprehensive_scores(self):
        """output comprehensive scores for a company's tweeting behavior pre vs. post series A."""
        scores = {}
        scores['all_tweet_num'] = len(self.tweets)
        scores['preA_tweet_num'] = len(self.preA_tweets)
        scores['postA_tweet_num'] = len(self.postA_tweets)
        scores['preA_tweet_freq'] = self.get_tweet_freq(self.preA_tweets, self.preA_timespan)
        scores['postA_tweet_freq'] = self.get_tweet_freq(self.postA_tweets, self.postA_timespan)
        scores['preA_tweet_content_richness'] = self.get_tweet_content_richness(self.preA_tweets)
        scores['postA_tweet_content_richness'] = self.get_tweet_content_richness(self.postA_tweets)
        scores['preA_tweet_interactiveness'] = self.get_tweet_interactiveness(self.preA_tweets)
        scores['postA_tweet_interactiveness'] = self.get_tweet_interactiveness(self.postA_tweets)
        return scores

def main():
    for i, row in companies_series_b_tweeted.iterrows():
        try:
            CT = CompanyTweet(row.twitter_username)
            print(i, row.twitter_username, row.first2last_funding_days, CT.comprehensive_scores())
        except:
            print(i, row.twitter_username, row.first2last_funding_days, 'ERROR')
    # print(example.comprehensive_scores())
main()