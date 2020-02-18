# Venturescope: Predicting startup's survival from their tweets
**Venturescope** is a predictive and analytical tool for venture capital investors to obtain data-driven insights about early-stage startups. This is a product from my Insight Data Science project (Jan 2020 cohort, Toronto).<br></br>
Website: www.datascipro.me.

## Problem
Venture capital is a high-risk, high-reward industry. Early-stage startups has failure rate of >95% after Seed-round funding, yet the successful ones become unicorn companies (e.g. FaceBook, Twitter, Alibaba), and generates 100x returns. It therefore becomes vital for VC investors to pick out the promising startups and avoid ones with high risk in their portfolio management. <br></br>
Since >50% of startups still fails at Series A (first funding round after Seed-round funding) as of now, this is not an easy job. The reason for high failure rate is that it is difficult to tell the "good" from the "bad" early-stage startups:

1. Early-stage startups usually have very limited disclosed information about their business and finance.
2. An average VC screens 15 companies daily, and the time to research each of them is limited.

As a result, when it comes to selection of startups for Series A funding, VC often relies on experience and references instead of data-driven insights, which leads to a less informed investment-decision-making process.

## Solution
**Venturescope** is a web application that leverages a startup's recent tweets to predict its Series A funding outcome. The prediction is based on insights from historical startup data retrived from Crunchbase and Twitter. Realizing that there is a strong correlation between a starup's tweets and its Series A outcome, I built a predictive model using a variety of machine learning methods (*Topic Modeling, Sentiment Analysis, Random Forest, Gradient Boosting*), and deployed it online to deliver these insights. Ultimately, **Venturescope** provides data-driven insights about startups, and saves venture capitalists time when screening startups for Series A investment.

## Demo
[![Venturescope Demo](https://img.youtube.com/vi/GCW9pZDV7TA/0.jpg)](https://www.youtube.com/watch?v=GCW9pZDV7TA&feature=youtu.be)

## Setting up Venturescope
After cloning this repository, run the following code:
```
cd Venturescope
pip3 install -r requirements.txt
```
This repository contains scripts, notebooks and Flask App source code. Due to storage space limit, raw data is not uploaded. (If you want to play with raw data yourself, send me an email.) Cleaned data is available in the `venturescope/venturescope/data/` directory. You can explore `scripts/` and `notebooks/` directory to check out my EDA, feature engineering, and model training!

## How Does Venturescope Work?
### Workflow
![workflow](https://github.com/haocai1992/insight_project/blob/master/notebooks/figures/workflow.jpg)
### Data
The raw data for this project has two parts: Crunchbase 2014 snapshot dataset, and scraped tweets from Twitter.<br></br>
The Crunchbase 2014 snapshot dataset was obtained from [Crunchbase Data](https://data.crunchbase.com/docs), which has the following information for ~3,000 startups that (i) went through Seed-round funding; (ii) either succeeded or failed in Series A funding; and (iii) has active Twitter account up until 2014:  
* country
* category
* market
* founded date
* founders
* \# of employees
* previous funding rounds
* previous funding amount

I also scraped ~600,000 tweets for ~3,000 startups in Crunchbase dataset using [TwitterScraper](https://github.com/taspinar/twitterscraper). The scraped tweets include following information:  
* tweet text
* \# of tweets
* date/time of tweets
* links
* hashtags
* images
* retweets
* replies
* likes

### Feature Engineering
#### 1) Numerical Features
Numerical features from Crunchbase and Twitter engineered were engineered either by counting or timescale calculation, including:
* \# of days since founded date
* \# of days since seed round
* \# of tweet since seed round
* frequency of tweeting
* average length of tweets
* tweet "content richness"
	- \# of links of each tweet
	- \# of hashtags of each tweet
	- \# of images of each tweet
* tweet "engagement score"
	- \# of likes of each tweet
	- \# of retweets of each tweet
	- \# of replies of each tweet
#### 2) NLP Features
A large part of this dataset is unstructured text (e.g. tweet text, industry description of the startup). In order to convert them into numerical features, I used multiple methods in natural language processing (Bag-of-words, TF-IDF topic modeling, Word2Vec and Sentiment Analysis), and engineered the following features:
* industry score
* market score
* tweet "topic score"
* tweet "sentiment score"

Details about feature engineering can be found in `notebooks/week3_Feature_Engineering.ipynb` .

### Model Training and Validation

Three binary classification models, Logistic Regression, Random Forest and Gradient Boosting were trained and cross-validated.
|Model|Precision|Recall|F1-score|
|-----|---------|------|--------|
|DummyClassifier(baseline)|0.524541|0.523810|0.524169|
|LogisticRegression|0.710642|0.664069|0.666178|
|RandomForestClassifier|0.749342|0.751515|0.744928|
|GradientBoostingClassifier|0.810110|0.811255|**0.810377**|


The highest F1-score is 0.81 from Gradient Boosting. Confusion Matrix and AUC curve (compared to other models) for GBClassifier:<br>
![confusion matrix and roc curve](https://github.com/haocai1992/insight_project/blob/master/notebooks/figures/confusion_matrix_and_roc_curve.png)
<br>
More than 80% of the success/failure were predicted using this model. This means that in Series A funding, more than 30% of failed VC investments could have been avoided!

## Limits and Future Directions
Venturescope was built during a short time period (3 weeks) as part of Insight Data Science program. Currently, it is not capable do real-time tweet retriving and analysis for starups. It also gives poor prediction results for startups that don't have Twitter accounts (which is very common in certain industries, e.g. Healthcare). Future improvements include:
* Real-time online tweet analysis of any given startup
* Engineering and incorporation of people-related features in the model
  * investor features
    - name of investor
    - investor relationships
    - previous investment outcomes
  * founder features
    - academic/professional background
    - \# of publications/patents/companies
    - previous entrepreneurship experience
* Visualization of model insights using BI tools (Tableau).

## Contact
* **Author**: Hao Cai
* **Email**: haocai3@gmail.com
* **Linkedin**: https://www.linkedin.com/in/haocai1992/
