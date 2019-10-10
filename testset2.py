import tweepy
from tweepy import OAuthHandler
import csv #Import csv
import re
consumer_key = ""
consumer_secret = "" 
access_token = ""
access_token_secret = "" 

auth = OAuthHandler(consumer_key, consumer_secret)  
auth.set_access_token(access_token, access_token_secret) 
api = tweepy.API(auth, wait_on_rate_limit=True)

tweetDataFile = "tweets.csv"
testquery = '#HUDMUN OR #ARSBHA OR #CHEWAT OR #MCILEI'

# Open/create a file to append data to
csvFile = open(tweetDataFile, 'a')

#Use csv writer
csvWriter = csv.writer(csvFile)
def clean_tweet(tweet): 
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split()) 
for tweet in tweepy.Cursor(api.search, q = testquery, lang = "en").items():
    # Write a row to the CSV file. I use encode UTF-8
    tweets = []
    tweets.append(tweet.text)
    if tweet.retweet_count > 0:
        if tweet.text not in tweets:
            csvWriter.writerow([clean_tweet(tweet.text.encode('utf-8')), "positive"])
    else:
	csvWriter.writerow([clean_tweet(tweet.text.encode('utf-8')), "positive"])
    print(clean_tweet(tweet.text))
csvFile.close()


