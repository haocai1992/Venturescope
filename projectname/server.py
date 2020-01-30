from flask import Flask, render_template, request
from projectname.custom_funcs import use_predictor

# Create the application object
app = Flask(__name__)

@app.route('/',methods=["GET","POST"]) #we are now using these methods to get user input
def home_page():
    return render_template('index.html')  # render a template

@app.route('/output')
def recommendation_output():
#    
    # Pull input
    company_name=request.args.get('user_input')

    # Case if empty
    if company_name =="":
        return render_template("index.html",\
                        my_input = company_name,\
                        my_form_result="Empty")
    else:
        estimate_time = use_predictor(company_name)
        some_number = 300
        some_image= "panda.gif"
        return render_template("index.html",\
                        my_input=company_name,\
                        my_output=estimate_time,\
                        my_number=some_number,\
                        my_img_name=some_image,\
                        my_form_result="NotEmpty")


# start the server with the 'run()' method
if __name__ == "__main__":
    app.run(debug=True) #will run locally http://127.0.0.1:5000/

