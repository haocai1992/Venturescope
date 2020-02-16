"""This script analyzes tweets topics using NLP, and engineers a "tweet topic score"."""

import pickle

import gensim
import numpy as np
np.random.seed(2018)
import pandas as pd
from gensim.utils import simple_preprocess
from gensim import corpora, models
from gensim.parsing.preprocessing import STOPWORDS
additional_stopwords = ['http', 'lnkd', 'https', 'html']
stopwords = set(STOPWORDS).union(additional_stopwords)
from nltk.stem import WordNetLemmatizer, SnowballStemmer
stemmer = SnowballStemmer('english')
from nltk.stem.porter import *

from data.config import raw_data_dir, processed_data_dir

pos_tweets = pd.read_csv(processed_data_dir + '/tweets_positive.csv').rename(columns={'0':'tweets'})\
               .sample(frac=0.1).copy()
neg_tweets = pd.read_csv(processed_data_dir + '/tweets_negative.csv').rename(columns={'0':'tweets'})\
               .sample(frac=0.1).copy()
all_tweets = pd.read_csv(processed_data_dir + '/tweets_all.csv').rename(columns={'0':'tweets'})

def lemmatize_stemming(text):
    """Lemmatize and stems text."""
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    """Preprocess text."""
    result = []
    for token in simple_preprocess(text):
        if token not in stopwords and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


class TwitterTopic:
    """A class to do topic modeling on tweets."""
    def __init__(self, tweets_df):
        self.tweets = tweets_df
        self.dictionary = None
        self.bow_corpus = None
        self.tfidf_model = None
        self.lda_model = None
        self.lda_model_tfidf = None

        try:
            self.preprocess_tweets()
            self.dictionary = self.get_dictionary()
            self.bow_corpus = self.get_doc2bow()
            self.tfidf_model = self.make_tfidf()
            self.lda_model = self.make_lda()
            self.lda_model_tfidf = self.make_lda_tfidf()
        except:
            print("ERROR performing topic modeling!")

    def preprocess_tweets(self):
        """methods to perform lemmatize and stem preprocessing steps on the data set."""
        self.tweets['docs'] = self.tweets['tweets'].map(preprocess)
        return None

    def get_dictionary(self):
        """Create a dictionary from ‘processed_docs’ containing the number\
         of times a word appears in the training set."""
        dictionary = corpora.Dictionary(self.tweets.docs)
        dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
        return dictionary

    def get_doc2bow(self):
        """Gensim doc2bow."""
        bow_corpus = [self.dictionary.doc2bow(doc) for doc in self.tweets.docs]
        return bow_corpus

    def make_tfidf(self):
        """Create tf-idf model object using models."""
        tfidf_model = models.TfidfModel(self.bow_corpus)
        return tfidf_model

    def make_lda(self):
        """Create LDA model object."""
        lda_model = models.LdaMulticore(self.bow_corpus, num_topics=10,
                                               id2word=self.dictionary, passes=2, workers=2)
        return lda_model

    def make_lda_tfidf(self):
        """Running LDA using TF-IDF."""
        lda_model_tfidf = models.LdaMulticore(self.tfidf_model[self.bow_corpus], num_topics=20,
                                                     id2word=self.dictionary, passes=2, workers=4)
        return lda_model_tfidf

    def match_new_tweets(self, text):
        """use LDA model to calculate new tweets' topic score"""
        # build similarity index.
        sims = gensim.similarities.Similarity(raw_data_dir+'/', self.tfidf_model[self.bow_corpus],
                                              num_features=len(self.dictionary))
        query_doc_bow = self.dictionary.doc2bow(preprocess(text))
        query_doc_tf_idf = self.tfidf_model[query_doc_bow]
        similarities = sims[query_doc_tf_idf]
        return np.mean(similarities)

def print_topics():
    """print the positive and negative topics."""
    pos_TT = TwitterTopic(pos_tweets)
    pos_topics = pos_TT.lda_model.print_topics()
    with open(processed_data_dir + '/tweetstopics_positive.pkl', 'wb') as f1:
        pickle.dump(pos_topics, f1)

    neg_TT = TwitterTopic(neg_tweets)
    neg_topics = neg_TT.lda_model.print_topics()
    with open(processed_data_dir + '/tweetstopics_negative.pkl', 'wb') as f2:
        pickle.dump(neg_topics, f2)
    return None

def write_tweet_topic_score(all_company_tweets):
    """Write tweet topic score for all companies to a file."""
    pos_TT = TwitterTopic(pos_tweets)
    neg_TT = TwitterTopic(neg_tweets)

    f = open(processed_data_dir + '/companies_tweets_topic_scores_sampled.txt', 'w')
    for i, row in all_company_tweets.iterrows():
        text = row['tweets']
        pos_tweets_similarity, neg_tweets_similarity = 0.0, 0.0
        try:
            pos_tweets_similarity = pos_TT.match_new_tweets(str(text))
            neg_tweets_similarity = neg_TT.match_new_tweets(str(text))
        except:
            print('ERROR calculating tweet topic similarity!')
        print("{:<10} {:<20.8} {:<20.8}".format(i, pos_tweets_similarity, neg_tweets_similarity))
        f.write("{:<10} {:<20.8} {:<20.8}\n".format(i, pos_tweets_similarity, neg_tweets_similarity))
    f.close()
    return
