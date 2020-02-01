"""Functions needed to run the prediction models."""
# from data.config import home_dir, raw_data_dir, processed_data_dir, cleaned_data_dir
from projectname.config import home_dir, raw_data_dir, processed_data_dir, cleaned_data_dir
import pandas as pd
import numpy as np
from scipy import stats
# from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

from nltk.stem import WordNetLemmatizer, SnowballStemmer
from gensim.utils import simple_preprocess
from gensim import corpora, models, similarities

from nltk.sentiment.vader import SentimentIntensityAnalyzer

all_companies = pd.read_csv(processed_data_dir + '/companies_all_labeled_final.csv')
usernames = all_companies.twitter_username
category_counts = pd.read_csv(processed_data_dir + '/category_counts.csv')
stopwords = pickle.load(open(processed_data_dir+'/stopwords.pkl', 'rb'))
stemmer = SnowballStemmer('english')

pos_tweets_dict = corpora.Dictionary().load(processed_data_dir + '/tweets_positive_dict.pkl')
neg_tweets_dict = corpora.Dictionary().load(processed_data_dir + '/tweets_negative_dict.pkl')
pos_tweets_tfidf_model = models.TfidfModel.load(processed_data_dir + '/tfidf_model_pos_tweets.model')
neg_tweets_tfidf_model = models.TfidfModel.load(processed_data_dir + '/tfidf_model_neg_tweets.model')
pos_tweets_sims = similarities.Similarity.load(processed_data_dir + '/tweets_positive_sims.pkl')
neg_tweets_sims = similarities.Similarity.load(processed_data_dir + '/tweets_negative_sims.pkl')

sid = SentimentIntensityAnalyzer()

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in simple_preprocess(text):
        if token not in stopwords and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

def match_new_tweets(tweet_text, tweets_sims, tweets_tfidf_model, tweets_dictionary):
    """calculate similarity score of new tweet to existing tweets (pos/neg)"""
    # build similarity index.
    sims = tweets_sims
    query_doc_bow = tweets_dictionary.doc2bow(preprocess(tweet_text))
    query_doc_tf_idf = tweets_tfidf_model[query_doc_bow]

    similarities_ = sims[query_doc_tf_idf]
    # print(np.mean(similarities_))
    return np.mean(similarities_)

class InputReader:
    """A class to convert user's input to numerical data for Predictor."""
    def __init__(self, input_values):
        self.companyname = str(input_values[0])
        self.countryname = str(input_values[1])
        self.companycategory = str(input_values[2])
        self.companyage = float(input_values[3])
        self.fundingage = float(input_values[4])
        self.tweetnum = int(input_values[5])
        self.inputtweet = str(input_values[6])
        self.inputtweetnum = int(input_values[7])

        self.datadict = {'country_feature': None,
                         'age_feature': None,
                         'days_since_first_funding': None,
                         'category_score': None,
                         'market_score': None,
                         'postA_tweet_num': None,
                         'postA_tweet_freq': None,
                         'postA_tweet_length': None,
                         'postA_tweet_content_richness': None,
                         'postA_tweet_interactiveness': None,
                         'tweets_pos_topic_score': None,
                         'tweets_neg_topic_score': None,
                         'tweets_neg_sent_score': None,
                         'tweets_neu_sent_score': None,
                         'tweets_pos_sent_score': None,
                         'tweets_compound_sent_score': None}
        try:
            self.convert_input_values()
        except:
            pass

    def convert_input_values(self):
        self.datadict['country_feature'] = 1 if self.countryname == 'USA' else 0
        self.datadict['age_feature'] = float(self.companyage)
        self.datadict['days_since_first_funding'] = float(self.fundingage * 365)
        self.datadict['category_score'] = self.cal_category_score(str(self.companycategory))
        self.datadict['market_score'] = all_companies['market_score'].mean() # use mean value.
        self.datadict['postA_tweet_num'] = self.tweetnum
        self.datadict['postA_tweet_freq'] = self.infer_tweet_freq(self.tweetnum)
        self.datadict['postA_tweet_length'] = self.digest_input_tweet(self.inputtweet, self.inputtweetnum)\
                                                        ['length']
        self.datadict['postA_tweet_content_richness'] = all_companies['postA_tweet_content_richness'].mean()
        self.datadict['postA_tweet_interactiveness'] = self.digest_input_tweet(self.inputtweet, self.inputtweetnum)\
                                                        ['interactiveness']
        self.datadict['tweets_pos_topic_score'] = self.digest_input_tweet(self.inputtweet, self.inputtweetnum)\
                                                        ['pos_topic_score']
        self.datadict['tweets_neg_topic_score'] = self.digest_input_tweet(self.inputtweet, self.inputtweetnum)\
                                                        ['neg_topic_score']
        self.datadict['tweets_neg_sent_score'] = self.digest_input_tweet(self.inputtweet, self.inputtweetnum) \
                                                        ['neg_sent_score']
        self.datadict['tweets_neu_sent_score'] = self.digest_input_tweet(self.inputtweet, self.inputtweetnum) \
                                                        ['neu_sent_score']
        self.datadict['tweets_pos_sent_score'] = self.digest_input_tweet(self.inputtweet, self.inputtweetnum) \
                                                        ['pos_sent_score']
        self.datadict['tweets_compound_sent_score'] = self.digest_input_tweet(self.inputtweet, self.inputtweetnum) \
                                                        ['compound_sent_score']
        return None

    @staticmethod
    def cal_category_score(category_name):
        corpus = category_counts.key.tolist()
        corpus.append(category_name)
        corpus_weight = category_counts.pos_minus_neg_count.values
        tfidf = TfidfVectorizer().fit_transform(corpus)
        pairwise_similarity = (tfidf * tfidf.T).toarray()[-1, :-1]
        score_sklearn = np.multiply(corpus_weight, pairwise_similarity).sum()
        return score_sklearn

    @staticmethod
    def infer_tweet_freq(tweet_num):
        """use the slope value from my feature engineering."""
        return float(0.003925651055633885 * tweet_num)

    @staticmethod
    def digest_input_tweet(inputtweet, inputtweetnum):
        """infers tweet scores from input tweets."""
        tweet_scores = {}
        tweet_scores['length'] = float(len(inputtweet)/int(inputtweetnum))
        tweet_scores['interactiveness'] = float((inputtweet.count('reply') +
                                                 inputtweet.count('replied') +
                                                 inputtweet.count('retweet'))/int(inputtweetnum))
        tweet_scores['pos_topic_score'] = match_new_tweets(tweet_text=inputtweet,
                                                           tweets_sims=pos_tweets_sims,
                                                           tweets_tfidf_model=pos_tweets_tfidf_model,
                                                           tweets_dictionary=pos_tweets_dict)
        tweet_scores['neg_topic_score'] = match_new_tweets(tweet_text=inputtweet,
                                                           tweets_sims=neg_tweets_sims,
                                                           tweets_tfidf_model=neg_tweets_tfidf_model,
                                                           tweets_dictionary=neg_tweets_dict)
        tweet_scores['neg_sent_score'] = sid.polarity_scores(inputtweet)['neg']
        tweet_scores['neu_sent_score'] = sid.polarity_scores(inputtweet)['neu']
        tweet_scores['pos_sent_score'] = sid.polarity_scores(inputtweet)['pos']
        tweet_scores['compound_sent_score'] = sid.polarity_scores(inputtweet)['compound']
        return tweet_scores


class Predictor:
    """A predictor to predict whether or not a company will raise another series of funding."""
    def __init__(self, model_file, data_file):
        self.model = self.load_model(model_file)
        self.data = self.load_data(data_file)

    @staticmethod
    def load_model(model_file):
        """Function to load a model."""
        return pickle.load(open(model_file, 'rb'))

    @staticmethod
    def load_data(data_file):
        """Function to load data."""
        return pd.read_csv(data_file)

    def predict(self, company_name):
        """Predict the class of time length of a company in the training set."""
        if company_name not in usernames.tolist():
            return "This company is not included in database yet!"
        else:
            company_data = self.data[usernames == company_name].values[:, :-1]
            predicted_proba = self.model.predict_proba(company_data)[0][1]
            return predicted_proba

def use_predictor_1(input_values):
    """func for Flask App call."""
    p = Predictor(model_file=processed_data_dir + '/model_xgb_2.pkl',
                  data_file=processed_data_dir + '/model_lr3_rf2_xgb2_data1.csv',)
    IR = InputReader(input_values)

    proba, other_info, stats_ = '', '', {}

    if IR.companyname in usernames.tolist():
        company_data = p.data[usernames == IR.companyname].values[:, :-1]
        if p.data[usernames == IR.companyname].values[:, -1][0] == 1.0:
            other_info += 'This company has already raised another round of funding!'
        if p.data[usernames == IR.companyname].values[:, -1][0] == 0.0:
            other_info += 'This company did not raise another round of funding!'
    else:
        # print(IR.datadict)
        company_data = pd.Series(IR.datadict).values.reshape((1, 16))

    proba += str(p.model.predict_proba(company_data)[0][1])

    stats_['category_ranking'] = stats.percentileofscore(p.data.category_score.values,
                                                        IR.datadict['category_score'])
    stats_['tweet_number_ranking'] = stats.percentileofscore(p.data.postA_tweet_num.values,
                                                            IR.datadict['postA_tweet_num'])
    stats_['tweet_length_ranking'] = stats.percentileofscore(p.data.postA_tweet_avglength.values,
                                                            IR.datadict['postA_tweet_length'])
    stats_['tweet_interactiveness_ranking'] = stats.percentileofscore(p.data.postA_tweet_interactiveness.values,
                                                            IR.datadict['postA_tweet_interactiveness'])
    topic_ranking = np.mean([stats.percentileofscore(p.data.tweets_pos_topic_score.values,
                                                    IR.datadict['tweets_pos_topic_score']),
                            stats.percentileofscore(p.data.tweets_neg_topic_score.values,
                                                    IR.datadict['tweets_neg_topic_score'])])
    stats_['tweets_topic_ranking'] = topic_ranking
    # sentiment_ranking = np.mean([stats.percentileofscore(p.data.tweets_neg_sent_score.values,
    #                                                 IR.datadict['tweets_neg_sent_score']),
    #                             stats.percentileofscore(p.data.tweets_neu_sent_score.values,
    #                                                     IR.datadict['tweets_neu_sent_score']),
    #                             stats.percentileofscore(p.data.tweets_pos_sent_score.values,
    #                                                     IR.datadict['tweets_pos_sent_score']),
    #                             stats.percentileofscore(p.data.tweets_compound_sent_score.values,
    #                                                 IR.datadict['tweets_compound_sent_score'])])
    stats_['tweets_sentiment_ranking'] = IR.datadict['tweets_pos_sent_score']
    # print(proba, other_info, stats_)
    return proba, other_info, stats_

def main():
    """test modules."""
    print(usernames.head())
    # use_predictor_1(input_values=['blah', 'USA', 'retail', '2', '1', '100', 'retail ecommerce', '1'])
    use_predictor_1(input_values=['2crisk', 'USA', 'retail', '2', '1', '100', 'retail ecommerce', '1'])

# main()