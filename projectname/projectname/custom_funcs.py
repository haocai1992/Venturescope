"""Functions needed to run the prediction models."""
# from data.config import home_dir, raw_data_dir, processed_data_dir, cleaned_data_dir
from projectname.config import home_dir, raw_data_dir, processed_data_dir, cleaned_data_dir
from projectname.config import classlabel2timelength
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle


class Predictor:
    """A predictor to predict the time length for a company to raise another series of funding."""
    def __init__(self, model_file, data_file):
        self.model = self.load_model(model_file)
        self.data = self.load_data(data_file)

    @staticmethod
    def load_model(model_file):
        """Function to load a model."""
        return pickle.load(open(model_file, 'rb'))

    @staticmethod
    def load_data(data_file):
        """Function to load data."""
        return pd.read_csv(data_file)

    def predict(self, company_name):
        """Predict the class of time length of a company in the training set."""
        if company_name not in self.data.name.tolist():
            return "This company is not included in database yet!"
        else:
            company_data = self.data[self.data.name == company_name].values[:, :-3]
            predicted_len_class = self.model.predict(company_data)[0]
            return classlabel2timelength[predicted_len_class]

def use_predictor(company_name):
    """func for Flask App call."""
    p = Predictor(model_file=processed_data_dir + '/model_logreg1.pkl',
                  data_file=processed_data_dir + '/model_logreg1_data.csv',)
    return p.predict(company_name=company_name)

# print(use_predictor('#waywire'))
