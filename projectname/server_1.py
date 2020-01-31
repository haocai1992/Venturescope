from flask import Flask, render_template, request, jsonify
from projectname.custom_funcs_1 import use_predictor_1
import numpy as np

# Create the application object
app = Flask(__name__)

@app.route('/') #we are now using these methods to get user input
def home_page():
    return render_template('index_1.html')  # render a template

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    temp = '.'.join([str(x) for x in request.form.values()])
    input_values = [x for x in request.form.values()]
    companyname = str(input_values[0])
    countryname = str(input_values[1])
    companycategory = str(input_values[2])
    companyage = str(input_values[3])
    fundingage = str(input_values[4])
    tweetnum = str(input_values[5])
    inputtweet = str(input_values[6])
    inputtweetnum = str(input_values[7])

    proba = float(use_predictor_1(input_values)[0])
    other_info = use_predictor_1(input_values)[1]
    stats_ = use_predictor_1(input_values)[2]

    output_proba = '{:<35} {:.4f}'.format('Probability of Next-round Funding', proba)
    output_other_info = str(other_info)
    output_stats = ['{:<35} {:.4f}'.format('company category score: ', float(stats_['category_ranking']/100.0)),
                    '{:<35} {:.4f}'.format('tweet number score: ', float(stats_['tweet_number_ranking']/100.0)),
                    '{:<35} {:.4f}'.format('tweet length score: ', float(stats_['tweet_length_ranking']/100.0)),
                    '{:<35} {:.4f}'.format('tweet interactiveness score: ', float(stats_['tweet_interactiveness_ranking']/100.0)),
                    '{:<35} {:.4f}'.format('tweet topic score: ', float(stats_['tweets_topic_ranking']/100.0)),
                    '{:<35} {:.4f}'.format('tweet sentiment score: ', float(stats_['tweets_sentiment_ranking']/100.0))]

    return render_template('index_1.html',
                           prediction_text=output_proba,
                           other_info=output_other_info,
                           category_ranking=output_stats[0],
                           tweet_number_ranking=output_stats[1],
                           tweet_length_ranking=output_stats[2],
                           tweet_interactiveness_ranking=output_stats[3],
                           tweet_topic_ranking=output_stats[4],
                           tweet_sentiment_ranking=output_stats[5])

# start the server with the 'run()' method
if __name__ == "__main__":
    app.run(debug=True) #will run locally http://127.0.0.1:5000/

