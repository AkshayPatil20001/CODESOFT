from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from gtts import gTTS
from tempfile import NamedTemporaryFile
import os

app = Flask(__name__)

# Load data
data = pd.read_csv('creditcard.csv')

# Separate legitimate and fraudulent transactions
legit = data[data['Class'] == 0]
fraud = data[data['Class'] == 1]

# Undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# Split data into features and target
X = data.drop(columns="Class", axis=1)
y = data["Class"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Train logistic regression model
model = LogisticRegression(solver='newton-cg')
model.fit(X_train, y_train)

# Evaluate model performance
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

# Function to convert text to speech and return audio file
def generate_audio(text):
    tts = gTTS(text=text, lang='en')
    audio_file = NamedTemporaryFile(delete=False)
    tts.save(audio_file.name)
    return audio_file.name

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_features = np.array(request.form['features'].replace('"', '').split(','), dtype=np.float64)[:30]
        if len(input_features) == 30:
            prediction = model.predict(input_features.reshape(1, -1))
            if prediction[0] == 0:
                audio_path = generate_audio("Legitimate transaction")
            else:
                audio_path = generate_audio("Fraudulent transaction")
            return render_template('index.html', prediction=prediction[0], audio_path=os.path.basename(audio_path))
        else:
            return render_template('index.html', error="Please provide the first 30 features.")
    return render_template('index.html')

@app.route('/play_audio/<path:audio_path>')
def play_audio(audio_path):
    return render_template('audio.html', audio_path=audio_path)

if __name__ == '__main__':
    app.run(debug=True)
