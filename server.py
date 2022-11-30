from flask import Flask, render_template 
from dotenv import load_dotenv 
from pic_fr import retrain_model 

load_dotenv()  # take environment variables from .env.

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    retrain_model()