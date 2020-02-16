"""Functions needed to run the prediction models."""

import pickle

import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy import stats

from venturescope.venturescope.config import data_dir

all_companies = pd.read_csv(data_dir + '/companies_all_labeled_final.csv')
usernames = all_companies.twitter_username
category_counts = pd.read_csv(data_dir + '/category_counts.csv')
model=pickle.load(open(data_dir + '/model_xgb_2.pkl', 'rb'))
model_data=pd.read_csv(data_dir + '/model_lr3_rf2_xgb2_data1.csv')

def predict_success(company_name):
    """Predict the chance of a company to raise series A funding."""
    if company_name not in usernames.tolist():
        print('ERROR WITH COMPANY NAME')
        return None
    else:
        company_data = model_data[usernames == company_name].values[:, :-1]
        predicted_proba = model.predict_proba(company_data)[0][1]
        return predicted_proba

def analyze_category(category_name):
    """Analyze the statistics of a company's category."""
    category_stats = {category_name: {'category_pct': None, 'success_pct': None}}
    if category_name not in category_counts.key.tolist():
        print('ERROR WITH CATEGORY NAME')
        return None
    else:
        category_total_count = category_counts.loc[category_counts.key==category_name, 'total_count'].values[0]
        total_success_count = category_counts.positive_count.sum()
        category_success_count = category_counts.loc[category_counts.key==category_name, 'positive_count'].values[0]

        category_pct = category_success_count/total_success_count # % of category in all successful startups
        success_pct = category_success_count/category_total_count # % of successful startups in this category

        category_stats[category_name]['category_pct'] = category_pct
        category_stats[category_name]['success_pct'] = success_pct
        return category_stats

def analyze_tweets(company_name):
    """Analyze the statistics of a company's tweets."""
    if company_name not in usernames.tolist():
        print('ERROR WITH TWEETS')
        return None
    else:
        tweets_stats = {"tweet_number_ranking": None,
                        "tweet_length_ranking": None,
                        "tweet_engagement_ranking": None,
                        "tweets_topic_ranking": None,
                        "tweets_sentiment_ranking": None,}

        tweet_num = model_data.loc[usernames==company_name, 'postA_tweet_num'].values[0]
        tweet_len = model_data.loc[usernames==company_name, 'postA_tweet_avglength'].values[0]
        tweet_engage_score = model_data.loc[usernames==company_name, 'postA_tweet_interactiveness'].values[0]
        tweet_topic_score = model_data.loc[usernames==company_name, 'tweets_pos_topic_score'].values[0]
        tweet_sent_score = model_data.loc[usernames==company_name, 'tweets_pos_sent_score'].values[0]

        tweets_stats["tweet_number_ranking"] = stats.percentileofscore(model_data.postA_tweet_num.values, tweet_num)
        tweets_stats["tweet_length_ranking"] = stats.percentileofscore(model_data.postA_tweet_avglength, tweet_len)
        tweets_stats["tweet_engagement_ranking"] = stats.percentileofscore(model_data.postA_tweet_interactiveness, tweet_engage_score)
        tweets_stats["tweets_topic_ranking"] = stats.percentileofscore(model_data.tweets_pos_topic_score, tweet_topic_score)
        tweets_stats["tweets_sentiment_ranking"] = stats.percentileofscore(model_data.tweets_pos_sent_score, tweet_sent_score)

        return tweets_stats

def create_plot(predicted_proba, category_stats, tweets_stats):
    """plot for prediction results."""
    # convert output numbers to plot input lists.
    name_success = ["success", "failure"]
    name_industry = [list(category_stats.keys())[0], 'others']
    success_pcts_startup = [predicted_proba, 1.0-predicted_proba]
    success_pcts_industry = [category_stats[name_industry[0]]['success_pct'],\
                             1.0-category_stats[name_industry[0]]['success_pct']]
    industry_pcts = [category_stats[name_industry[0]]['category_pct'],\
                     1.0-category_stats[name_industry[0]]['category_pct']]

    x_tweet_stats = [tweets_stats["tweets_sentiment_ranking"],\
                     tweets_stats["tweets_topic_ranking"],\
                     tweets_stats["tweet_engagement_ranking"],\
                     tweets_stats["tweet_length_ranking"],\
                     tweets_stats["tweet_number_ranking"]]
    y_tweet_statnames = ['tweet sentiment', 'tweet topic', 'tweet engagement', 'length of tweets', 'number of tweets', ]

    # make plots.
    fig = make_subplots(rows=2,
                        cols=3,
                        specs=[[{"type": "pie"}, {"type": "pie"}, {"type": "pie"}],\
                               [{"type": "bar", "rowspan": 1,"colspan": 3}, None, None]],

                        subplot_titles=["this startup's <br> chance of success<br>",
                                        'overall chance of success<br>in {} industry'.format(list(category_stats.keys())[0]),\
                                        "proportion of successful startups <br> in {} industry"\
                                        .format(list(category_stats.keys())[0]),
                                        "This Startup's Twitter Performance",\
                                        '',\
                                        ''],)
                        # subplot_titles=["This startup's <br>Success Chance",\
                        #                 "This Startup's Twitter Performance",\
                        #                 'Success Chance <br>in {} Industry'.format(list(category_stats.keys())[0]),\
                        #                 'Proportion of {} Industry in all starups'\
                        #                 .format(list(category_stats.keys())[0]),])

    # pie plot: "This startup's Success Chance"
    fig.add_trace(go.Pie(name = "",
                         values = success_pcts_startup,
                         labels = ['success', 'failure'],
                         hovertext = ['The chance of {} for this startup is {:.1%}'\
                                .format(name_success[i], success_pcts_startup[i]) for i in range(len(name_success))],
                         text = ['success', 'failure'],
                         textposition='inside',
                         marker=dict(colors=['#1f77b4','red']),
                         hoverinfo = 'text',),
                 row=1, col=1)

    # pie plot: 'Industry-average Success Chance'
    fig.add_trace(go.Pie(name = "",
                         values = success_pcts_industry,
                         labels = ['success', 'failure'],
                         hovertext = ['The chance of {} for all startups in this industry is {:.1%}'\
                                .format(name_success[i], success_pcts_industry[i]) for i in range(len(name_success))],
                         text = ['success', 'failure'],
                         textposition='inside',
                         marker=dict(colors=['#1f77b4','red']),
                         hoverinfo = 'text',),
                 row=1, col=2)

    # pie plot: 'Industry Proportion'
    fig.add_trace(go.Pie(name = "",
                         values = industry_pcts,
                         labels = name_industry,
                         hovertext = ['{1:.1%} of successful starups are from \"{0}\" industry'\
                                     .format(name_industry[i], industry_pcts[i]) for i in range(len(name_industry))],
                         text = name_industry,
                         textposition='inside',
                         marker=dict(colors=['green','lightblue']),
                         hoverinfo = 'text',),
                 row=1, col=3)

    # bar plot: 'Twitter Performance'
    fig.add_trace(go.Bar(x=x_tweet_stats,
                         y=y_tweet_statnames,
                         hovertext = ['This startup is doing better than {:.1%} companies in {}'\
                                    .format(x_tweet_stats[i]/100.0, y_tweet_statnames[i]) for i in range(len(x_tweet_stats))],
                         orientation='h',
                         marker=dict(color='blue', line=dict(color='white', width=1)),
                         hoverinfo='text'),
                 row=2, col=1)

    fig.add_trace(go.Bar(x=[100-x_ for x_ in x_tweet_stats],
                         y=y_tweet_statnames,
                         orientation='h',
                         marker_color='lightgray',
                         hoverinfo='skip'),
                  row=2, col=1)

    fig.update_layout(barmode='stack',
                      autosize=True,
                      showlegend=False,
                      xaxis={'title':'percentile ranking',},)

    graphJSON = fig.to_json()
    # graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

def use_predictor(company_name, category_name='test'):
    """func for Flask App call."""
    results = {'head_info': None, 'category_info': None, 'tweet_info': None, 'plot': None}

    company_name_lower, category_name_lower = company_name.lower(), category_name.lower()

    prob = predict_success(company_name_lower)
    category_stats = analyze_category(category_name_lower)
    tweets_stats = analyze_tweets(company_name_lower)

    # if prob is None:
    #     results['head_info'] = 'This startup is not in our database yet... Please check again in a few weeks!'
    # else:
    #     results['head_info'] = "This startup's chance to raise series A runding is: {:.1%}".format(prob)

    # if category_stats is None:
    #     results['category_info'] = "This startup's industry is not in our database yet... Please check again in a few weeks!"
    # else:
    #     results['category_info'] = "This startup's industry accounts for {category_pct:.1%} of all startups; {success_pct:.1%} of starups in this industry raised series A funding!"\
    #                                 .format(**category_stats[category_name])

    # if tweets_stats is None:
    #     results['tweet_info'] = "This startup's tweets are not in our database yet... Please check again in a few weeks!"
    # else:
    #     results['tweet_info'] = "This startup's Twitter performance rankings are:\n\
    #     Number - {tweet_number_ranking:.1f}%\n\
    #     Length - {tweet_length_ranking:.1f}%\n\
    #     Engagement - {tweet_engagement_ranking:.1f}%\n\
    #     Topic - {tweets_topic_ranking:.1f}%\n\
    #     Sentiment - {tweets_sentiment_ranking:.1f}%".format(**tweets_stats)

    if (prob is None) or (tweets_stats is None):
        results['head_info'] = 'This startup is not in our database yet... Please check again in a few weeks!'

    elif (category_stats is None):
        category_list = all_companies.loc[all_companies.twitter_username == company_name_lower, "category_list"].values[0].split('|')
        if category_list[1].lower() in category_counts.key.tolist():
            category_stats = analyze_category(category_list[1].lower())
        else:
            results['head_info'] = "This startup's category is not in our database yet... Please check again in a few weeks!"

    if (prob is not None) and (category_stats is not None) and (tweets_stats is not None):
        if prob >= 0.5:
            results['head_info'] = 'Prediction result for {}: This startup will raise series A funding!'.format(company_name)
        else:
            results['head_info'] = "Prediction result for {}: This startup won't raise series A funding!".format(company_name)

        results['plot'] = create_plot(predicted_proba=prob, category_stats=category_stats, tweets_stats=tweets_stats)

    return results

