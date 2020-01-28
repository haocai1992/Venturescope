from data.config import raw_data_dir, processed_data_dir, cleaned_data_dir, tweets_data_dir
import spacy
from gensim.models import KeyedVectors

import pandas as pd
import pickle

nlp = spacy.load("en_core_web_md")
gensim_model = KeyedVectors.load_word2vec_format(raw_data_dir + '/GoogleNews-vectors-negative300.bin', binary=True)
category_counts = pd.read_csv(processed_data_dir + '/category_counts.csv')
market_counts = pd.read_csv(processed_data_dir + '/market_counts.csv')

def word_similarity_spacy_eng(word1, word2):
    if nlp(word1).vector_norm and nlp(word2).vector_norm:
        return nlp(word1).similarity(nlp(word2))
    else:
        return 0.0

def word_similarity_gensim_google(word1, word2):
    if word1 in gensim_model and word2 in gensim_model:
        return gensim_model.similarity(word1, word2)
    else:
        return 0.0

def word_score(word, word_counts_df):
    word_simlarities_spacy = []
    word_simlarities_gensim = []
    score_spacy = 0.0
    score_gensim = 0.0
    for i, row in word_counts_df.iterrows():
        word_ = row['key']
        positive_count = row['positive_count']
        negative_count = row['negative_count']
        total_count = row['total_count']
        pos_minus_neg_count = positive_count - negative_count
        sim_spacy_eng = word_similarity_spacy_eng(word, word_)
        sim_gensim_google = word_similarity_spacy_eng(word, word_)
        word_simlarities_spacy.append(sim_spacy_eng)
        word_simlarities_gensim.append(sim_gensim_google)
        score_spacy += float(sim_spacy_eng) * float(pos_minus_neg_count)
        score_gensim += float(sim_gensim_google) * float(pos_minus_neg_count)
    word_counts_df['similarity_spacy_eng'] = pd.Series(word_simlarities_spacy)
    word_counts_df['similarity_gensim_google'] = pd.Series(word_simlarities_gensim)

    return score_spacy

print(word_score('music', category_counts[:10]))
