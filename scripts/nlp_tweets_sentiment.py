"""This script analyzes tweets sentiments using NLP, and engineers a "tweet sentiment score"."""

# import nltk
# nltk.download('vader_lexicon')
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

from analyze_tweets import CompanyTweet
from data.config import processed_data_dir
from nlp_tweets_topic import lemmatize_stemming, preprocess

all_companies = pd.read_csv(processed_data_dir + '/companies_all_labeled.csv')

def preprocess_sentence(text):
    """Preprocess sentences (stemming and lemmatization followed by joining words)."""
    sentence = ' '.join(preprocess(text))
    return sentence

class TweetSentiment(CompanyTweet):
    """A class to analyze tweet sentiment."""
    def __init__(self, twitter_username):
        CompanyTweet.__init__(self, twitter_username)
        self.selected_tweets = None
        self.processed_tweets = None
        self.sentiments = None
        self.sentiment_score = {'neg':0.0, 'neu':0.0, 'pos':0.0, 'compound':0.0}
        try:
            self.selected_tweets = self.select_tweets()
            self.processed_tweets = self.preprocess_tweets()
            self.sentiments = self.analyze_sentiment(self.processed_tweets)
            self.sentiment_score = self.calculate_sentiment_score()
        except:
            print('ERROR calculating sentiment score!')

    def select_tweets(self):
        """select tweets to do sentiment analysis on. For now, look at first 100 tweets post series A."""
        selected_tweets = self.postA_tweets.text[:100].copy()
        return selected_tweets

    def preprocess_tweets(self):
        """methods to perform lemmatize and stem preprocessing steps on the data set."""
        processed_tweets = self.selected_tweets.map(preprocess_sentence)
        return processed_tweets

    @staticmethod
    def analyze_sentiment(tweets_text):
        """analyze the tweet sentiments."""
        sentiment_scores = {'neg':[], 'neu':[], 'pos':[], 'compound':[]}
        for tweet in tweets_text:
            score = sid.polarity_scores(tweet)
            sentiment_scores['neg'].append(score['neg'])
            sentiment_scores['neu'].append(score['neu'])
            sentiment_scores['pos'].append(score['pos'])
            sentiment_scores['compound'].append(score['compound'])
        sentiment_df = pd.DataFrame(sentiment_scores)
        return sentiment_df

    def calculate_sentiment_score(self):
        """return a score dict for all sentiments."""
        score = self.sentiments.mean().to_dict()
        return score

def write_sentiment_score():
    """write sentiment scores for all companies to a file."""
    f = open(processed_data_dir + '/companies_tweets_sentiment_scores.txt', 'w')
    for i, row in all_companies.iterrows():
        username = row.twitter_username
        neg_score, neu_score, pos_score, compound_score = 0.0, 0.0, 0.0, 0.0
        try:
            TS = TweetSentiment(username)
            s_score = TS.sentiment_score
            neg_score = s_score['neg']
            neu_score = s_score['neu']
            pos_score = s_score['pos']
            compound_score = s_score['compound']
        except:
            print("ERROR calculating sentiment score!")

        print("{:<10} {:<25} {:<15.8} {:<15.8} {:<15.8} {:<15.8}"\
              .format(i, username, neg_score, neu_score, pos_score, compound_score))
        f.write("{:<10} {:<25} {:<15.8} {:<15.8} {:<15.8} {:<15.8}\n"\
              .format(i, username, neg_score, neu_score, pos_score, compound_score))
    f.close()
    return
