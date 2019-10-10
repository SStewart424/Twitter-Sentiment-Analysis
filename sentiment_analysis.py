import pandas as pd
import re, nltk, csv
import collections
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import ngrams
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
count_vectorizer = CountVectorizer(ngram_range=(1,2))

stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()

column_names=['tweet','sentiment']
tweets = pd.read_csv("tweetDataFile.csv", header=None,names= column_names)
sentiment_counts = tweets.sentiment.value_counts()
number_of_tweets = tweets.tweet.count()

def normalizer(tweet):
    only_letters = re.sub("[^a-zA-Z]", " ",tweet) 
    tokens = nltk.word_tokenize(only_letters)[2:]
    lower_case = [l.lower() for l in tokens]
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
    return lemmas

def ngrams(input_list):
    #onegrams = input_list
    bigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:]))]
    trigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:], input_list[2:]))]
    return bigrams+trigrams

def count_words(input):
    cnt = collections.Counter()
    for row in input:
        for word in row:
            cnt[word] += 1
    return cnt

def sentiment2target(sentiment):
    return {
        'negative': 0,
        'positive' : 1
    }[sentiment]
targets = tweets.sentiment.apply(sentiment2target)

pd.set_option('display.max_colwidth', -1)
tweets['normalized_tweet'] = tweets.tweet.apply(normalizer)
tweets[['tweet','normalized_tweet']].head()
tweets['grams'] = tweets.normalized_tweet.apply(ngrams)

vectorized_data = count_vectorizer.fit_transform(tweets.tweet)
indexed_data = hstack((np.array(range(0,vectorized_data.shape[0]))[:,None], vectorized_data))

data_train, data_test, targets_train, targets_test = train_test_split(indexed_data, targets, test_size=0.1, random_state=0)
data_train_index = data_train[:,0]
data_train = data_train[:,1:]
data_test_index = data_test[:,0]
data_test = data_test[:,1:]
# for NB
NaiveBayes = MultinomialNB()
NaiveBayes_output = NaiveBayes.fit(data_train, targets_train)

print(NaiveBayes.score(data_test, targets_test))
column =['tweet']
tweets2 = pd.read_csv("tweets.csv", header=None, names=column)
sentences = count_vectorizer.transform(tweets2.tweet)
predictions = NaiveBayes.predict(sentences)
negative=0
positive=0
for prob in predictions:
    if (prob==0):
	negative+=1
    else:
	positive+=1
print(100*negative/(negative+positive))
# For SVM
SupVecMac = svm.SVC(gamma=0.01, C=100., probability=True, class_weight='balanced', kernel='linear')
SupVecMac_output = SupVecMac.fit(data_train, targets_train)

print(SupVecMac.score(data_test, targets_test))
column =['tweet']
tweets2 = pd.read_csv("tweets.csv", header=None, names=column)
sentences = count_vectorizer.transform(tweets2.tweet)
predictions = SupVecMac.predict(sentences)
negative=0
positive=0
for prob in predictions:
    if (prob==0):
	negative+=1
    else:
	positive+=1
print(100*negative/(negative+positive))



