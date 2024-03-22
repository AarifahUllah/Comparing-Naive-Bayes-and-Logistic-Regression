"""
Natural Language Understanding Homework 1: Comparing Naive Bayes to Logisitic Regression
Aarifah Ullah, Date: February 2, 2024
Prof. Kasia Hitczenko
CSCI 4907.82
"""
# Library Calls
import os
import spacy
from collections import Counter #word frequency counter
import numpy as np
import pandas as pd #read/writing csv file
from math import log
import string
from sklearn.feature_extraction.text import CountVectorizer #for bag of words, preprocessing tweets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB 
from sklearn.metrics import accuracy_score,f1_score
from sklearn.linear_model import LogisticRegression

#PART 1: PREPARING THE DATA

#LOADING DATA
df = pd.read_csv("Tweets_5K.csv") # read from csv
raw_tweets = df['text'].tolist() #extract list of un-processed tweets
labels = df['sentiment'].tolist() #corresponding list of sentiments (positive, negative, neutral)

#BAISC PREPROCESSING
basic_preproc_tweets = list() #initiliaze a list that holds words from basic preprocessing
for tweet in raw_tweets: #iterate through list of un-processed tweets
    text_block = tweet.split() #obtain individual words from an individual tweet
    for word in text_block:
        basic_preproc_tweets.append(word) #add individual word to list of basic processed tweets
#print (*basic_preproc_tweets, sep = " ")

#positive = +1, negative = -1, neutral = 0
labels_nums = list() # initialize a list that would hold the integer representations of sentiment scores
for sentiment in labels: #iterate through whole list (note: this would maintain the order)
    if sentiment == "positive":
        labels_nums.append(1) #add +1 
    elif sentiment == "negative":
        labels_nums.append(-1) #add -1
    elif sentiment == "neutral":
        labels_nums.append(0) #add 0

pos_count = labels_nums.count(1)
neg_count = labels_nums.count(-1)
neutral_count = labels_nums.count(0)
#print("positive docs:", pos_count, " negative docs:", neg_count,  " neutral count:", neutral_count)

#FEATURIZE (BAG OF WORDS)
vectorizer = CountVectorizer() #initialize an instance of count vectorizer
basic_preproc_bow = vectorizer.fit_transform(raw_tweets) #use the vectorizer to put the raw tweets in a document-term matrix
basic_array = basic_preproc_bow.toarray()
#basic_preproc_tweets = vectorizer.get_feature_names_out() #retreive all the words used in the tweets. (alphabetical order)
#print(basic_preproc_tweets.__getitem__(1460))

#PART 4: IMPLEMENTING MORE ELABORATE PRE-PROCESSING
"""
The order of pre-processing matters, as you may get unwanted results
w/o careful attention to what steps to apply first
1. removing punctuation & extra white space
2. lowercasing
3. replacing numbers with NUM
4. remove stop words first
5. lemmatization
6. then choose 1000 most frequent words & the rest with OOV. 
Imagine you chose the most freuent words first, then remove the stop words, you lose out on a lot of data.
"""

#removing punctuation
more_preproc_tweets = list()
translation = str.maketrans("", "", string.punctuation)
for word in raw_tweets:
    more_preproc_tweets.append(word.translate(translation))
#print(*more_preproc_tweets, sep="\n")

#lowercasing
lower_case_tweets = list()
for word in more_preproc_tweets:
    lower_case_tweets.append(word.lower())
#print(*lower_case_tweets, sep = "\n")

"""
#replacing numbers with NUM
no_nums_tweets = lower_case_tweets
for tweet in no_nums_tweets:
    for word in tweet:
        if word.isdigit():
            word = 'NUM'
print(*no_nums_tweets, sep= "\n")
"""

nlp = spacy.load("en_core_web_sm")

#removing stop words
no_stops_tweet = " "
processed_tweet = list()
all_stops_tweets = list()
for tweet in lower_case_tweets: #iterating through individual tweets
    tweet_ = nlp(tweet) #using spacy's library
    for word in tweet_:
        if (word.is_stop) or (str(word) == ['s', 't', 'm', 'd', 'z', 'e', 'nd', 'nt', 've', 'l', 'n', 'e']): #checking if a word is a stop word
            processed_tweet.append(str(""))
        else:
            processed_tweet.append(str(word))
    no_stops_tweet = no_stops_tweet.join(processed_tweet) #add individual words back together
    all_stops_tweets.append(no_stops_tweet) #add string to list of strings
    processed_tweet.clear() #clear processed_tweet list & 
    no_stops_tweet=" "#no_stops_tweet list for next iteration of tweets
#print(*all_stops_tweets, sep= "\n")

#lemmatization
root_tweet = " "
processing_tweet = list()
all_lemmatized_tweets = list() #list of lemmatized tweets
for tweet in all_stops_tweets: #iterating through individual tweets
    tweet_=nlp(tweet) #using spacy's library
    for word in tweet_:
        processing_tweet.append(str(word.lemma_))#retreive root word
    root_tweet = root_tweet.join(processing_tweet) #add the tweet back together
    all_lemmatized_tweets.append(root_tweet) #add to list of all root tweets
    processing_tweet.clear() #clear iterable list
    root_tweet = " "
#print(*all_lemmatized_tweets, sep="\n")

#most frequent 1000 words plus one
all_lemmatized_words = list() #hold the lemmatized words from tweets
most_frequent_words = list() #to hold the 1000 words
complete_proccessed_words = list() #hold list of 1000 words and OOV
complete_tweet = ""#space holder
complete_tweets = list() #final list of tweets
#retreive most common words from all lemmatized tweets:
for tweet in all_lemmatized_tweets:
    for word in tweet:
        all_lemmatized_words.append(word)
most_frequent = (Counter(all_lemmatized_words).most_common(1000)) #find most common words & counts
#print(all_lemmatized_words, sep="\n")
#print(most_frequent)
for i in most_frequent:
    most_frequent_words.append(i[0]) #add 1000 most common words into a list
#processed_words = set(all_lemmatized_words) #assign lemmatized tweets and 
common_words = set(most_frequent_words) #the most frequent words into sets
#print(common_words, sep="\n")

for tweet in all_lemmatized_tweets: #iterate through individual tweets
    for word in tweet: #iterate through the tweet's word
        if word in common_words: #check if the word in lemmatized tweets is a common word
            complete_proccessed_words.append(word)
        elif word not in common_words:
            complete_proccessed_words.append('OOV') #otherwise assign it as OOV
    complete_tweet = complete_tweet.join(complete_proccessed_words) #add the wordd back together
    complete_tweets.append(complete_tweet) #add to list of all completed tweets
    complete_proccessed_words.clear()
    complete_tweet = "" #reset iterables
#print(*complete_tweets, sep="\n")
vectorizer2 = CountVectorizer(max_features=1001, stop_words='english')
more_proc_bow = vectorizer2.fit_transform(complete_tweets)
final_proc_tweets = vectorizer2.get_feature_names_out() #retreive all the words used in the tweets. (alphabetical order)
#---------------------------------------

#INITIALIZING TRAINING SET AND TEST SET
#assign first 80% of data as the training set & the last 20% as the test set
X = more_proc_bow #input 2D array (matrix) use w/ more_proc_bow or w/ basic_preproc_bow
y = labels_nums #labels as an array of ints
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= .8, random_state= None, shuffle= False, stratify= None)

#PART 2: IMPLEMENTING NAIVE BAYES
naive_bayes_model = MultinomialNB() #build Multinomial Classifier
naive_bayes_model.fit(X_train, y_train) #model training

#testing on unseen data
y_prediction_naive = naive_bayes_model.predict(X_test)
accuracy_naive = accuracy_score(y_test, y_prediction_naive)
f1_naive = f1_score(y_prediction_naive, y_test, average = "weighted")

#print("Naive Accuracy:", accuracy_naive)  #pre-processing result: 0.54575
#print("NaiveF1 Score:", f1_naive) #pre-processing result: 0.5554424766889603

#-----------------------------------------------------
#PART 3: IMPLEMENTING LOGISTIC REGRESSION
logistic_regression_model = LogisticRegression(max_iter= 150).fit(X_train, y_train) #call the logistic regression class
y_predicition_logistic = logistic_regression_model.predict(X_test)
accuracy_logisitic = accuracy_score(y_test, y_predicition_logistic)
f1_logistic = f1_score(y_prediction_naive, y_test, average = "weighted")
indices = [i for i in range(len(y_test)) if y_test[i] != y_predicition_logistic[i]]
wrong_predictions = (df.iloc[indices,:])
#print(wrong_predictions)

#print("Logistic Accuracy:", accuracy_logisitic)  #pre-processing result: 0.58
#print("Logistic F1 Score:", f1_logistic)         #pre-processing result: 0.5554424766889603
#print(basic_preproc_bow.toarray())
#print(more_proc_bow.shape) #the dimensions of the feature matrix#PART 4: IMPLEMENTING MORE ELABORATE PRE-PROCESSING