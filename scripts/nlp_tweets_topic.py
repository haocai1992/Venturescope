"""This script analyzes tweets topics using NLP."""
from data.config import raw_data_dir, processed_data_dir, cleaned_data_dir, tweets_data_dir
import pandas as pd
from gensim.utils import simple_preprocess
from gensim.test.utils import get_tmpfile
from gensim import corpora, models, similarities
from gensim.parsing.preprocessing import STOPWORDS
additional_stopwords = ['http', 'lnkd', 'https', 'html']
stopwords = set(STOPWORDS).union(additional_stopwords)

from nltk.stem import WordNetLemmatizer, SnowballStemmer
stemmer = SnowballStemmer('english')
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
# nltk.download('wordnet')
import pickle

# pos_tweets = pd.read_csv(processed_data_dir + '/tweets_positive.csv').rename(columns={'0':'tweets'})
# print(pos_tweets.head())
# neg_tweets = pd.read_csv(processed_data_dir + '/tweets_negative.csv').rename(columns={'0':'tweets'})
# print(neg_tweets.head())
# all_tweets = pd.read_csv(processed_data_dir + '/tweets_all.csv').rename(columns={'0':'tweets'})
# # print(all_tweets.count())
# pos_tweets_docs = pickle.load(open(processed_data_dir + '/tweets_positive_docs.pkl', 'rb'))
# neg_tweets_docs = pickle.load(open(processed_data_dir + '/tweets_negative_docs.pkl', 'rb'))

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in simple_preprocess(text):
        if token not in stopwords and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

def get_dictionary(docs):
    dictionary = corpora.Dictionary(docs)
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    return dictionary

def get_doc2bow(dict, docs):
    """Gensim doc2bow."""
    bow_corpus = [dict.doc2bow(doc) for doc in docs]
    print(bow_corpus)
    return bow_corpus

pos_tweets_dict = corpora.Dictionary().load(processed_data_dir + '/tweets_positive_dict.pkl')
neg_tweets_dict = corpora.Dictionary().load(processed_data_dir + '/tweets_negative_dict.pkl')

pos_tweets_bow_corpus = pickle.load(open(processed_data_dir + '/tweets_positive_bow_corpus.pkl', 'rb'))
neg_tweets_bow_corpus = pickle.load(open(processed_data_dir + '/tweets_negative_bow_corpus.pkl', 'rb'))

pos_tweets_tfidf_model = models.TfidfModel.load(processed_data_dir + '/tfidf_model_pos_tweets.model')
neg_tweets_tfidf_model = models.TfidfModel.load(processed_data_dir + '/tfidf_model_neg_tweets.model')

# index_temp = get_tmpfile("index")
# pos_tweets_sims = similarities.Similarity(output_prefix=index_temp,
#                                           corpus=pos_tweets_tfidf_model[pos_tweets_bow_corpus],
#                                           num_features=len(pos_tweets_dict))
# neg_tweets_sims = similarities.Similarity(output_prefix=index_temp,
#                                           corpus=neg_tweets_tfidf_model[neg_tweets_bow_corpus],
#                                           num_features=len(neg_tweets_dict))
# pos_tweets_sims.save(processed_data_dir + '/tweets_positive_sims.pkl')
# neg_tweets_sims.save(processed_data_dir + '/tweets_negative_sims.pkl')
pos_tweets_sims = similarities.Similarity.load(processed_data_dir + '/tweets_positive_sims.pkl')
neg_tweets_sims = similarities.Similarity.load(processed_data_dir + '/tweets_negative_sims.pkl')

def match_new_tweets(tweet_text, tweets_sims, tweets_tfidf_model, tweets_dictionary):
    """calculate similarity score of new tweet to existing tweets (pos/neg)"""
    # build similarity index.
    sims = tweets_sims
    query_doc_bow = tweets_dictionary.doc2bow(preprocess(tweet_text))
    query_doc_tf_idf = tweets_tfidf_model[query_doc_bow]

    similarities_ = sims[query_doc_tf_idf]
    # print(np.mean(similarities_))
    return np.mean(similarities_)


class TwitterTopic:
    """A class to do topic modeling on tweets."""
    def __init__(self, tweets_df):
        self.tweets = tweets_df
        self.dictionary = None
        self.bow_corpus = None
        self.tfidf_model = None
        self.lda_model = None
        self.lda_model_tfidf = None

        print('TRY')
        self.preprocess_tweets()
        self.dictionary = self.get_dictionary()
        self.bow_corpus = self.get_doc2bow()
        self.tfidf_model = self.make_tfidf()
        self.lda_model = self.make_lda()
        # self.lda_model_tfidf = self.make_lda_tfidf()
        try:
            print('TRY')
            self.preprocess_tweets()
            self.dictionary = self.get_dictionary()
            self.bow_corpus = self.get_doc2bow()
            self.tfidf_model = self.make_tfidf()
            self.lda_model = self.make_lda()
            # self.lda_model_tfidf = self.make_lda_tfidf()
        except:
            pass

    def preprocess_tweets(self):
        """methods to perform lemmatize and stem preprocessing steps on the data set."""
        self.tweets['docs'] = self.tweets['tweets'].map(preprocess)
        return None

    def get_dictionary(self):
        """Create a dictionary from ‘processed_docs’ containing the number\
         of times a word appears in the training set."""
        dictionary = gensim.corpora.Dictionary(self.tweets.docs)
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
        lda_model = gensim.models.LdaMulticore(self.bow_corpus, num_topics=10,
                                               id2word=self.dictionary, passes=2, workers=2)
        return lda_model

    def make_lda_tfidf(self):
        """Running LDA using TF-IDF."""
        lda_model_tfidf = gensim.models.LdaMulticore(self.tfidf_model[self.bow_corpus], num_topics=20,
                                                     id2word=self.dictionary, passes=2, workers=4)
        return lda_model_tfidf

    def match_new_tweets(self, text):
        """use LDA model to calculate new tweets' topic"""
        # build similarity index.
        sims = gensim.similarities.Similarity(raw_data_dir+'/', self.tfidf_model[self.bow_corpus],
                                              num_features=len(self.dictionary))
        query_doc_bow = self.dictionary.doc2bow(preprocess(text))
        query_doc_tf_idf = self.tfidf_model[query_doc_bow]

        similarities = sims[query_doc_tf_idf]
        return np.mean(similarities)

def print_topics():
    """print the positive and negative topics."""
    # print('TEST1')
    # pos_TT = TwitterTopic(pos_tweets)
    # pos_topics = pos_TT.lda_model.print_topics()
    # print(pos_topics)
    # with open(processed_data_dir + '/tweetstopics_positive.pkl', 'wb') as f1:
    #     pickle.dump(pos_topics, f1)

    pos_topics = pickle.load(open(processed_data_dir + '/tweetstopics_positive.pkl', 'rb'))
    print('pos', pos_topics)

    # print('TEST2')
    # neg_TT = TwitterTopic(neg_tweets)
    # neg_topics = neg_TT.lda_model.print_topics()
    # print(neg_topics)
    # with open(processed_data_dir + '/tweetstopics_negative.pkl', 'wb') as f2:
    #     pickle.dump(neg_topics, f2)

    neg_topics = pickle.load(open(processed_data_dir + '/tweetstopics_negative.pkl', 'rb'))
    print('neg', neg_topics)
    return None

def get_tweet_topic_score(all_company_tweets):
    print('TEST1')
    pos_TT = TwitterTopic(pos_tweets)
    neg_TT = TwitterTopic(neg_tweets)
    print('TEST2')

    f = open(processed_data_dir + '/companies_tweets_topic_scores.txt', 'w')
    for i, row in all_company_tweets.iterrows():
        text = row['tweets']
        pos_tweets_similarity, neg_tweets_similarity = 0.0, 0.0
        try:
            pos_tweets_similarity = pos_TT.match_new_tweets(text)
            neg_tweets_similarity = neg_TT.match_new_tweets(text)
        except:
            pass
        print("{:<10} {:<20.8} {:<20.8}".format(i, pos_tweets_similarity, neg_tweets_similarity))
        f.write("{:<10} {:<20.8} {:<20.8}\n".format(i, pos_tweets_similarity, neg_tweets_similarity))
    f.close()
    return


def main():
    # get_tweet_topic_score(all_tweets)
    # print_topics()
    pos_TT = TwitterTopic(pos_tweets)
    print(pos_TT.tweets['docs'])
    # with open(processed_data_dir + '/TwitterTopic_positive.pkl', 'wb') as f:
    #     pickle.dump(pos_TT, f)
    # pos_TT = pickle.load(open(processed_data_dir + '/TwitterTopic_positive.pkl', 'rb'))
    # print(pos_TT.match_new_tweets('blabla'))
# main()
