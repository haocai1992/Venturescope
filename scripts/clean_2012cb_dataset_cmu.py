"""This script cleans the 2012 crunchbase dataset (from CMU)."""

import os
import pandas as pd
from EntityParser import EntityParser

# data directories.
raw_data_dir = '/Users/caihao/Dropbox/Insight_Jan2020/datasets/cmu_crunchbase_data'
companies_dir = raw_data_dir + '/company'
articles_dir = raw_data_dir + '/article'
cleaned_data_dir = '/Users/caihao/PycharmProjects/insight-project/data/cleaned'

# clean companies and store in dataframe.
companies = {'name': [], 'overview': [], 'description':[], 'category_code':[],
             'permalink': [], 'crunchbase_url': [], 'homepage_url': [], 'blog_url': [],
             'blog_feed_url': [], 'twitter_username':[], 'number_of_employees':[], 'founded_year':[],
             'founded_month': [], 'founded_day':[], 'competitions':[], 'total_money_raised':[],
             'funding_rounds': [], 'investments': [], 'acquisition': [], 'acquisitions': [],
             'milestones': [], 'ipo': [], 'tag_list': []}

for company in os.listdir(companies_dir):
    print("=============")
    js = EntityParser.LoadJsonEntity(companies_dir + '/' + company)
    if js:
        for key in companies:
            if key in js:
                print(key, js[key])
                companies[key].append(js[key])
            else:
                companies[key].append(None)
                print(key, None)
    else:
        print("no json object")

df = pd.DataFrame(companies)
# df.to_csv(cleaned_data_dir + '/' + 'cmu_cb_company.csv', index=False)

