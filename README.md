# Venturescope
A web app that predicts the startup's series A funding success based on their recent tweets. This is my Insight Data Science project (Jan 2020 cohort). The web app is here: www.datascipro.me.

## Motivation
Venture capital is a high-risk, high-reward industry. In 2008, VC investors invested in more than 1000 US startups. 10 of them are still around today. Of the 99% that failed, half of them failed at early stage (from seed round to series A). This is because at this stage, it’s very hard for VC to make fully-informed decisions, partly because don’t have time to perform due diligence for so many startups, and also that they don’t have a lot of accurate info about each startup. So they often have to rely on their experience and instincts sometimes, which leads to high failure rate.

## Solution
Using supervised machine learning approach can reduce this investment risk by leveraging hisorical data to predict whether a seed round startup will raise series A funding or not.

## Data
I have two data sources (CrunchBase and Twitter). Crunchbase is the largest database for startups that VCs use to acquire information such as a startup’s country, category, market, founded date, founders, employees, previous funding amount and rounds. The problem with Crunchbase data is that it is very basic and doesn’t contain a lot of business details about a company. Also, it doesn't update very often, so it has a lot of obsolete information as well.

That’s why I am using Twitter as a supplementary data source. Early-startups often use Twitter to increase their publicity among customers and investors. Key features I extracted from startup’s tweets include tweets’ number, frequency and average tweet length. I also engineered two numerical features from tweets, “content richness” and “engagement score”. “Content richness” is a feature that summarizes number of links, hashtags and images of tweets, and “engagement score” summarizes number of likes, retweets and replies of tweets.

Crunchbase dataset is obtained from Crunchbase dataset, which contains ~3000 startups' information until 2014. Twitter data was obtained by scraping tweets during the period where each startup went through (or not) their series A funding.

## Algorithm
Three classification models, Logistic Regression, Random Forest and Gradient Boosting was used for the classification performance comparison. A natural language processing pipeline was built using Word2Vec, Topic Modeling and Sentimental Analysis during feature engineering process, in order to convert text into numerical features to feed to the classifier.

## Outcomes
The highest F1-score from Gradient Boosting is 0.81. The most important features identified were “Tweet Topic Score”, “Tweet Engagement Score” and “Tweet Content Richness”. 

## Challenges and limits
placeholder placeholder placeholder placeholder placeholder placeholder placeholder placeholder placeholder placeholder placeholder placeholder placeholder placeholder placeholder placeholder 

## Future
placeholder placeholder placeholder placeholder placeholder placeholder placeholder placeholder placeholder placeholder placeholder placeholder placeholder placeholder placeholder placeholder 
