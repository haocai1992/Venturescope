from flask import Flask
app = Flask(__name__)
from venturescope.venturescope import server
