from nltk.sentiment import SentimentIntensityAnalyzer
import bs4
import requests
import re
import streamlit as st
#import streamlit.components.v1 as components
import nltk
import pandas as pd
import pickle
import os.path
import sklearn
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from aitextgen import aitextgen

nltk.download('vader_lexicon')

def web_scraper_critics(movie):
    critic_url = 'http://www.rottentomatoes.com/m/'+movie + '/reviews?type=top_critics'
    #audience_url = 'https://www.rottentomatoes.com/m/' + movie + '/reviews?type=user'
    response = requests.get(critic_url)
    soup = bs4.BeautifulSoup(response.text)
    reviews = soup.find_all('div', {'class': 'the_review'})
    if len(reviews) == 0:
        return []
    cleaned_reviews = [
        re.findall('[^\r\n]+', i.text) if len(re.findall('[^\r\n]+', i.text)) == 0 else re.findall('[^\r\n]+', i.text)[
            0].strip() for i in reviews]
    df = pd.DataFrame({'review': [cleaned_reviews]}, index = [0])
    df.to_csv('cleaned_reviews_critics.csv')
    return


def web_scraper_audience(movie):
    st.success("YO")
    audience_url = 'https://www.rottentomatoes.com/m/' + movie + '/reviews?type=user'
    response = requests.get(audience_url)
    soup = bs4.BeautifulSoup(response.text)
    reviews = soup.find_all('div', {'class': 'audience-reviews__review-wrap'})
    print(reviews)
    if len(reviews) == 0:
        return []
    cleaned_reviews = [i.find('p', {'data-qa': 'review-text'}).text for i in reviews]
    df = pd.DataFrame({'review': [cleaned_reviews]}, index= [0])
    df.to_csv('cleaned_reviews_audience.csv')
    return


def sentiment_analysis(review_list):
    # Sentiment analyzer (VADER)
    # nltk.download('vader_lexicon')
    sa = SentimentIntensityAnalyzer()

    sa_dict = sa.polarity_scores("I see says the blind man")

    # All of the audience reviews
    #df = pd.read_csv("audience_reviews.csv")

    reviews = review_list.split("', ")
    #reviews = review_list
    # Iterate through all of the reviews and calculate average sentiment
    #for index, row in df.iterrows():
    sent_score_comp = []
    sent_score_neu = []
    sent_score_pos = []
    sent_score_neg = []

    for i in reviews:


        sent_score_comp.append(sa.polarity_scores(i)['compound'])
        sent_score_neu.append(sa.polarity_scores(i)['neu'])
        sent_score_pos.append(sa.polarity_scores(i)['pos'])
        sent_score_neg.append(sa.polarity_scores(i)['neg'])

    compound_score = sum(sent_score_comp) / len(sent_score_comp)
    positive_score = sum(sent_score_pos) / len(sent_score_pos)
    neutral_score = sum(sent_score_neu) / len(sent_score_neu)
    negative_score = sum(sent_score_neg) / len(sent_score_neg)



    # Add average sentiment as a new column to the dataframe
    return [compound_score, positive_score, neutral_score, negative_score]

@st.cache
def predictive():
    st.header("Rotten Tomatoes Audience Score Prediction")
    text = open("predictive.txt").read()
    st.write(text)
    movie = st.text_input('Insert movie name here:')
    if st.button('Scrape Critic Reviews'):
        web_scraper_critics(movie)
    if st.button('Scrape Audience Reviews'):
        web_scraper_audience(movie)
    if os.path.exists('cleaned_reviews_audience.csv'):
        audience_reviews = pd.read_csv('cleaned_reviews_audience.csv')['review'].iloc[0]
        if len(audience_reviews) == 0:
            st.success('There are no audience reviews for this movie')
            return
    if os.path.exists('cleaned_reviews_critics.csv'):
        critic_reviews = pd.read_csv('cleaned_reviews_critics.csv')['review'].iloc[0]

        if len(critic_reviews) == 0:
            st.success('There are no critic reviews for this movie')
            return
        
    model = pickle.load(open('model.pkl', 'rb'))
    if st.button('Predict Audience Score'):
        critics = sentiment_analysis(critic_reviews)
        st.markdown('Compound Score Critics:')
        st.markdown(critics[0])
        st.markdown('Positive Score Critics:')
        st.markdown(critics[1])
        st.markdown('Neutral Score Critics:')
        st.markdown(critics[2])
        st.markdown('Negative Score Critics:')
        st.markdown(critics[3])
        audience = sentiment_analysis(audience_reviews)
        st.markdown('Compound Score Audience:')
        st.markdown(audience[0])
        st.markdown('Positive Score Audience:')
        st.markdown(audience[1])
        st.markdown('Neutral Score Audience:')
        st.markdown(audience[2])
        st.markdown('Negative Score')
        st.markdown(audience[3])

        prediction = model.predict(pd.DataFrame({'compound_audience':audience[0], 'neutral_audience':audience[2], 'positive_audience':audience[1], 'negative_audience':audience[3], 'positive':critics[1], 'negative':critics[3], 'neutral':critics[2], 'compound':critics[0]}, index = [0]))
        st.success(prediction)


def sentiment():
    st.header("Sentiment Analysis")
    st.image("Images/sentiment-analysis.png")
    st.subheader("Background")
    st.markdown("Sentiment analysis is a technique that uses natural language processing and machine learning to "
                "analyze text and determine its relative positive, neutral, or negative sentiment scores. In our "
                "project, we utilized NLTK's pretrained sentiment analyzer known as VADER (Valence Aware Dictionary "
                "and sEntiment Reasoner). We inputted each review into the VADER analysis tool, and then retrieved "
                "the resulting sentiment scores. The resulting sentiment scores are grouped into neutral, positive, "
                "negative, and compound. Generally, a positive sentiment score indicates that the review thought "
                "positively about the movie, and a negative score indicates that the review did not think positively "
                "about the movie. Neutral indicates the amount of text that has no sentiment values, and compound "
                "combines all three of the other scores. Following this, we aggregated the data by combining all of the "
                "scores for each movie and then averaging the values. The result of this is displayed below:")

    st.subheader("Audience Movie Reviews Sentiment Scores")
    audience_sent = pd.read_csv("Data/audience_reviews_sentiment.csv")
    audience_sent[["Movies", "Reviews", "compound_sentiment", "neutral_sentiment",
                   "positive_sentiments", "negative_sentiments"]]

    st.subheader("Critic Movie Reviews Sentiment Scores")
    critic_sent = pd.read_csv("Data/critics_sentiment.csv",encoding = 'unicode_escape')
    critic_sent = critic_sent[["movie_name", "review", "positive", "negative", "neutral",
                               "compound", "decision"]].groupby("movie_name").mean()
    critic_sent = critic_sent.reset_index()
    critic_sent
    # st.markdown("aodsifnaosfjsdfjodados")
    # critic_sent.columns
    #
    # st.bar_chart(critic_sent)
    # fig = px.scatter(x=critic_sent["decision"], y=critic_sent["compound"])
    # fig.update_layout(xaxis_title="Decision", yaxis_title="Sentiment",)
    # st.write(fig)
    #
    # st.markdown('<style>body{background-color: Red;}</style>',unsafe_allow_html=True)

@st.cache
def text_gen():
    temperatures = [0.2, 0.5, 1.0, 1.2]

    st.header("Text Generator")

    description = 'Utilizing both GPT-2 as well as aitextgen, the text generation model is trained on the critic reviews from the top four most popular genres. The model then takes in a prompt inputted by the user and generates 100 characters of text. The prompt can be a movie title or a user written review. The model has a temperature parameter that essentially determines how conservative(lower values) or risky(higher values) the model\'s prediction will be. In the example, a phrase is inputted with varying temperature values. The lower valued temperature examples more closely resemble actual phrases used in the critic reviews while the higher values become more unintelligible.'

    st.write(description)
   
    ex = aitextgen(model_folder="comedy_trained_model",
                   tokenizer_file="comedy_aitextgen.tokenizer.json")

    st.text("Example:'Chicken Run is a good movie'")
    for temperature in temperatures:
        t = 'Temperature:' + str(temperature)
        st.text(t)
        gpt_text = ex.generate_one(prompt="Chicken Run is a good movie",
                                   max_length=100, temperature=temperature)
        st.write(gpt_text)

    option = st.selectbox('Select a Genre', ('comedy', 'drama', 'documentary', 'mystery'))
    st.write('You selected:', option)

    ai = aitextgen(model_folder=f"{option}_trained_model",
                   tokenizer_file=f"{option}_aitextgen.tokenizer.json")

    prompt_text = st.text_input(label="Enter your prompt:",
                                value=" ")

    if len(prompt_text) > 1:
        with st.spinner("Generating text..."):
            gpt_text = ai.generate_one(prompt=prompt_text,
                                       max_length=100, temperature=0.2)
        st.success("Successfully generated the text below! ")

        print(gpt_text)

        st.text(gpt_text)


def intro():
    st.image("Images/rtlogo.png")
    st.title("Rotten Tomatoes Data Analysis")
    st.header("Introduction")
    intro_text = open("intro.txt").read()
    st.write(intro_text)


def scraping():
    text = open("web_scraping_and_data.txt").read()
    st.header("Web Scraping and Data")
    st.write(text)
    df = pd.read_csv("all_movie_reviews.csv")
    df


def conclusion():
    text = open("conclusion.txt").read()
    st.header("Conclusion")
    st.write(text)


def modeling():
    text = open("modeling.txt").read()
    st.header("Topic Modeling")
    st.write(text)
    st.image("Images/Topic_Modeling_example.png")


def tfidf():
    text = open("tfidf.txt").read()
    st.header("TF-IDF")
    st.write(text)
    #html = open("topic_modelling_critics.html", 'r', encoding='utf-8')
    #source_code = html.read()
    #components.html(source_code, height=1200, width=1200)


intro()
scraping()
sentiment()
tfidf()
modeling()
predictive()
text_gen()
conclusion()
