import os

from flask import Flask, request, render_template, jsonify

from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax

application = app = Flask(__name__)

# Preprocess text
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def predict_sentiment(text):
  text = preprocess(text)
  encoded_input = tokenizer(text, return_tensors='pt')
  output = model(**encoded_input)
  scores = output[0][0].detach().numpy()
  scores = softmax(scores)
  ranking = np.argsort(scores)
  ranking = ranking[::-1]
  sentiment = config.id2label[ranking[0]] # one of "negative", "neutral", "positive"
  return sentiment

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
  text = request.form['text']
  sentiment = predict_sentiment(text)
  return render_template('index.html', result=sentiment)

if __name__ == '__main__':
  app.run()
