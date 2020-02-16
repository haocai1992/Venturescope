from flask import render_template, request
from venturescope_ import app

from venturescope.venturescope.custom_funcs import use_predictor


# Create the application object
# app = Flask(__name__)

@app.route('/') #we are now using these methods to get user input
def home_page():
    return render_template('index.html')  # render a template

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    input_values = [x for x in request.form.values()]
    company_name, country_name, category, age, funding_age = input_values

    if company_name == '':
      results = use_predictor(company_name= 'Spotify', category_name='music')

    else:
      results = use_predictor(company_name=company_name, category_name=category)

    return render_template('index.html',
                           head_info = results['head_info'],
                           # category_info = results['category_info'],
                           # tweet_info = results['tweet_info'],
                           plot=results['plot'],)

# start the server with the 'run()' method
# if __name__ == "__main__":
#    app.run(host='0.0.0.0', debug=True) #will run locally http://127.0.0.1:5000/

