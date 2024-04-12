# app.py
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

app = Flask(__name__)

# Load the model and vectorizer
with open('vectorizer.pkl', 'rb') as model_vectorizer_file:
    vectorizer = pickle.load(model_vectorizer_file)

# Load the model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
# Load the LabelEncoder
with open('label_encoder.pkl', 'rb') as le_file:
    label_encoder = pickle.load(le_file)

# Function to preprocess the input text
def preprocess_text(text):
    # Your preprocessing code here
    return text

# Function to predict whether the message is spam or ham
def predict_spam_ham(message):
    # Preprocess the message
    processed_message = preprocess_text(message)
    # Vectorize the processed message
    vectorized_message = vectorizer.transform([processed_message])
    # Predict the class
    prediction = model.predict(vectorized_message)[0]
    # Decode the prediction
    decoded_prediction = label_encoder.inverse_transform([prediction])[0]
    return decoded_prediction

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        # Predict whether the message is spam or ham
        prediction = predict_spam_ham(message)
        return render_template('result.html', message=message, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
