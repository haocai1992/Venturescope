import os
from EntityParser import EntityParser
import pandas as pd

# data directories.
data_dir = '/home/caihao/PycharmProjects/insight-project/data'
raw_data_dir = data_dir + '/raw'
processed_data_dir = data_dir + '/processed'
cleaned_data_dir = data_dir + '/cleaned'

# clean companies and related articles data.
companies_dir = raw_data_dir + '/companies'
articles_dir = raw_data_dir + '/articles'

companies = {'name': [], 'overview': [], 'description': [], 'category_code': []}
for company in os.listdir(companies_dir):
        js = EntityParser.LoadJsonEntity(companies_dir + '/' + company)
        if js:
            companies['name'].append(js['name'])
            companies['overview'].append(js['overview'])
            companies['description'].append(js['description'])
            companies['category_code'].append(js['category_code'])

articles = {}
for company in os.listdir(articles_dir):
    for article in os.listdir(articles_dir + '/{}'.format(company)):
        art_text = open(articles_dir + '/{}/{}'.format(company, article)).read()
        articles[company] = art_text

# companies['article_text'] = []
# for name in companies['name']:
#     if name in articles:
#         companies['article_text'].append(articles[name])
#     else:
#         companies['article_text'].append('')

df = pd.DataFrame(companies)
print(df.description)
# df.to_csv(cleaned_data_dir + '/companies_week1test.csv', index=False)
# print(pd.read_csv(cleaned_data_dir + '/companies_week1test.csv'))