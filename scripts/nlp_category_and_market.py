"""This script analyzes the 'category' and 'market' columns (text data), and engineers two numerical\
features (using NLP word2vec-clustering)."""

import re
import sys
sys.path.append("/Users/caihao/PycharmProjects/insight_project/")

import numpy as np
import pandas as pd
# import spacy
# nlp = spacy.load("en_core_web_md")
# from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer

from data.config import processed_data_dir

companies_df = pd.read_csv(processed_data_dir + '/companies_all_labeled.csv')
# category_counts = pd.read_csv(processed_data_dir + '/category_counts.csv')
# market_counts = pd.read_csv(processed_data_dir + '/market_counts.csv')
category_counts = pd.read_csv(processed_data_dir + '/category_counts_sampled.csv')
market_counts = pd.read_csv(processed_data_dir + '/market_counts_sampled.csv')

def word_similarity_spacy_eng(word1, word2):
    """Similary of two words using SpaCy English pretrained model."""
    if nlp(word1).vector_norm and nlp(word2).vector_norm:
        return nlp(word1).similarity(nlp(word2))
    else:
        return 0.0

def word_similarity_gensim_google(word1, word2):
    """Similary of two words using gensim Google pretrained model."""
    if word1 in gensim_model and word2 in gensim_model:
        return gensim_model.similarity(word1, word2)
    else:
        return 0.0

def word_score(word, word_count_dict):
    """Score (using SpaCy English model) for a word compared to word counts dict."""
    score_spacy = 0.0
    for word_, count in word_count_dict.items():
        similarity_spacy_eng = word_similarity_spacy_eng(word, word_)
        score_spacy += float(count) * float(similarity_spacy_eng)
    return score_spacy

def word_score_sklearn(word, word_count_df=category_counts):
    """Score (using sklearn.text) for a word compared to word counts dataframe. Reason to use this - Fast!"""
    corpus = word_count_df.key.tolist()
    corpus.append(word)
    corpus_weight = word_count_df.pos_minus_neg_count.values
    tfidf = TfidfVectorizer().fit_transform(corpus)
    pairwise_similarity = (tfidf * tfidf.T).toarray()[-1, :-1]
    score_sklearn = np.multiply(corpus_weight, pairwise_similarity).sum()
    return score_sklearn

def get_category_score(companies_df):
    """Add "category score" to the companies df."""
    category_count_dict = pd.Series(category_counts.pos_minus_neg_count.values, index=category_counts.key).to_dict()

    f = open(processed_data_dir + '/companies_all_category_scores_sampled.txt', 'w')
    for i, category_list in enumerate(companies_df.category_list.fillna('|').to_list()):
        categories = re.split('\||\+', category_list)
        score = 0.0
        try:
            for category in categories:
                if len(category) > 0:
                    # score += word_score(category, category_count_dict)
                    score += word_score_sklearn(category, word_count_df=category_counts)
        except:
            print("ERROR writing category score!")
        print('{:<20} {:<20.8f}'.format(i, score))
        f.write('{:<20} {:<20.8f}\n'.format(i, score))
    f.close()
    return

def get_market_score(companies_df):
    """add "market score" to the companies df."""
    market_count_dict = pd.Series(market_counts.pos_minus_neg_count.values, index=market_counts.key).to_dict()

    f = open(processed_data_dir + '/companies_all_market_scores_sampled.txt', 'w')
    for i, market_list in enumerate(companies_df.market.fillna('+').to_list()):
        markets = re.split('\+', market_list)
        score = 0.0
        try:
            for market in markets:
                if len(market) > 0:
                    # score += word_score(market, market_count_dict)
                    score += word_score_sklearn(market, word_count_df=market_counts)
        except:
            print("ERROR writing category score!")
        print('{:<20} {:<20.8f}'.format(i, score))
        f.write('{:<20} {:<20.8f}\n'.format(i, score))
    f.close()
    return

# get_category_score(companies_df)
# get_market_score(companies_df)