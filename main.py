import os.path

import streamlit as st

import dotenv
import nltk

nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

fold_path = dotenv.get_key('.env', 'FOLDER_PATH')


# creates dictionary with file names (without .txt extension) as keys and
# text from files as values
def get_text_from_files(folder_path):
    data = {}
    try:
        for name in os.listdir(folder_path):
            with open(os.path.join(folder_path, name), 'r', encoding='utf8') as file:
                txt = file.read()
                date = name.replace('.txt', '')
                data[date] = txt
    except Exception as e:
        print('Error occurred!', e)
    return data


# analyse text and find sentiment scores,
# create dictionary with dates (file names) as keys and
# sentiment scores as values,
# append dictionaries to list and return this list
def get_sentiment_scores(text_dict, positive_or_negative):
    analyzer = SentimentIntensityAnalyzer()
    scores = []
    for key, val in text_dict.items():
        if positive_or_negative == 'positive':
            scores.append({'date': key, 'sentiment_score': analyzer.polarity_scores(val)['pos']})
        if positive_or_negative == 'negative':
            scores.append({'date': key, 'sentiment_score': analyzer.polarity_scores(val)['neg']})
    return scores


text = get_text_from_files(fold_path)
sentiment_scores_positive = get_sentiment_scores(text, 'positive')
sentiment_scores_negative = get_sentiment_scores(text, 'negative')

figure, ax1 = plt.subplots()  # Create a figure containing a single Axes.

st.title('Analyze Diary')
st.subheader('Positivity')
x_label = ax1.set_xlabel('Dates')
y_label = ax1.set_ylabel('Positivity score')


def get_points(data_dict):
    x_pts = [el['date'] for el in data_dict]
    y_pts = [el['sentiment_score'] for el in data_dict]
    return [x_pts, y_pts]


positive_points = get_points(sentiment_scores_positive)
x_points_positive = positive_points[0]
y_points_positive = positive_points[1]
ax1.plot(x_points_positive, y_points_positive)  # Plot some data on the Axes.
st.pyplot(figure)

figure, ax2 = plt.subplots()
st.subheader('Negativity')
x_label = ax2.set_xlabel('Dates')
y_label = ax2.set_ylabel('Negativity score')
negative_points = get_points(sentiment_scores_negative)
x_points_negative = negative_points[0]
y_points_negative = negative_points[1]
ax2.plot(x_points_negative, y_points_negative)

st.pyplot(figure)
