import re 
from tweepy import OAuthHandler
import jsonpickle
import csv
import time
import nltk
import string
from textblob import TextBlob 
import pandas as pd

positive = 0
negative = 0
neutral = 0
column =['tweet']
column_names=['tweet','sentiment']
tweets2 = pd.read_csv("tweets.csv", header=None, names=column)
tweets= pd.read_csv("tweetDataFile.csv", header=None, names=column_names)

for tweet in tweets2.tweet:
    analysis = TextBlob(tweet) 
    if analysis.sentiment.polarity > 0: 
        positive+=1
    elif analysis.sentiment.polarity == 0: 
        neutral+=1
    else: 
        negative+=1

length = positive+negative+neutral
print("Positive Sentiment Percentage = " + str(100*positive/length) + "%")

print("Negative Sentiment Percentage = " + str(100*negative/length) + "%")
print(length)

