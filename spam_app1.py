import streamlit as st
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

st.title("Spam Message Detection ")
st.write("Enter a message ")

raw_data = pd.read_csv('final_spam_dataset (1).csv', encoding='latin-1')

raw_data = raw_data.dropna(subset=['message', 'label']).copy()
raw_data['message'] = raw_data['message'].astype(str)
raw_data['message'] = raw_data['message'].str.strip()

X = raw_data['message']
Y = raw_data['label']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)


model = LogisticRegression()
model.fit(X_train_features , Y_train
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)


input_message = st.text_area("Enter your message:")
if st.button("Predict"):
    if input_message:
        input_data_features = feature_extraction.transform([input_message])
        prediction = model.predict(input_data_features)
        if prediction[0] == 1:
            st.success("✅ Ham Message")
        else:
            st.error("🚨 Spam Message")
    else:
        st.warning("⚠️ Please enter a message.")

st.write("Model Test Accuracy:", round(accuracy_on_test_data * 100, 2), "%")