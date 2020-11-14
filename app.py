#!/usr/bin/env python
# coding: utf-8

# importing the required libraries
from flask import Flask, render_template, request, redirect, url_for
from joblib import load
import numpy as np

dprdetection = load("depressiondetection.joblib")

# start flask
app = Flask(__name__)

# render default webpage
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/', methods=['POST', 'GET'])
def get_data():
    if request.method == 'POST':
        text = request.form['search']
        prob_prediction = dprdetection.predict_proba([text])
        class_prediction = dprdetection.predict([text])
        prob = float(np.max(prob_prediction))
        result = int(class_prediction[0])
        some_text = "Probability: "
        if result == 1:
            text_positive = "Not depressed"
            #return "Not depressed<br>Probability: " + str(prob)
            return render_template('home.html', probability = prob, result_class = text_positive, def_text = some_text, original_text = text)
        else:
            text_negative = "Depressed"
            #return "Depressed<br>Probability: " + str(prob)
            return render_template('home.html', probability = prob, result_class = text_negative, def_text = some_text, original_text = text)


@app.route('/success/<text>')
def success(text):
    prob_prediction = dprdetection.predict_proba([text])
    class_prediction = dprdetection.predict([text])
    prob = float(np.max(prob_prediction))
    result = int(class_prediction[0])
    if result == 1:
        return "Not depressed<br>Probability: " + str(prob)
    else:
        return "Depressed<br>Probability: " + str(prob)

if __name__ == "__main__":
    app.run(debug=True)
