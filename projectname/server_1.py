from flask import Flask, render_template, request, jsonify
from projectname.custom_funcs import use_predictor
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

    output = temp

    return render_template('index_1.html', prediction_text='Employee Salary should be $ {}'.format(output))

# start the server with the 'run()' method
if __name__ == "__main__":
    app.run(debug=True) #will run locally http://127.0.0.1:5000/

