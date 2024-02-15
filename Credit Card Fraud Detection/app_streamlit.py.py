import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st
from gtts import gTTS
from tempfile import NamedTemporaryFile

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
model = LogisticRegression()
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

# Create Streamlit app
st.title("Credit Card Fraud Detection Model")
st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")

# Create input fields for the user to enter feature values
input_df = st.text_input('Input first 30 features (separated by commas)')
# Create a button to submit input and get prediction
submit = st.button("Submit")

if submit and input_df:
    # Get input feature values and preprocess
    input_features = np.array(input_df.replace('"', '').split(','), dtype=np.float64)[:30]
    # Check if number of features matches
    if len(input_features) == 30:
        # Make prediction
        prediction = model.predict(input_features.reshape(1, -1))
        # Display result with color
        if prediction[0] == 0:
            st.markdown('<p style="color:green;">Legitimate transaction</p>', unsafe_allow_html=True)
            audio_path = generate_audio("Legitimate transaction")
        else:
            st.markdown('<p style="color:red;">Fraudulent transaction</p>', unsafe_allow_html=True)
            audio_path = generate_audio("Fraudulent transaction")
        
        # Use HTML5 audio tag to play audio automatically
        st.audio(audio_path, format='audio/mp3', start_time=0)
    else:
        st.write("Please provide the first 30 features.")
elif submit:
    st.write("Please provide input features.")
