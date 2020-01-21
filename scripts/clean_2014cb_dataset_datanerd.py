"""This script cleans the 2014 crunchbase dataset (from datanerd)."""
import pandas as pd

cb_data_dir = '/Users/caihao/PycharmProjects/insight-project/data/raw/cb_2014_datanerd'

# read all companies.
companies = pd.read_csv(cb_data_dir + '/' + 'companies.csv')

# select ones that has only one funding round.
companies_1round = companies.loc[companies.funding_rounds==1]
companies_2round = companies.loc[companies.funding_rounds==2]
companies_3round = companies.loc[companies.funding_rounds>2]

print(companies.columns)
