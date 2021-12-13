! pip install kaggle
! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json

! kaggle datasets download gauravduttakiit/bitcoin-tweets-16m-tweets-with-sentiment-tagged
# ! kaggle datasets download alaix14/bitcoin-tweets-20160101-to-20190329

! unzip bitcoin-tweets-16m-tweets-with-sentiment-tagged.zip

import pandas as pd

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
link = 'https://drive.google.com/file/d/1eUxIBU4oq5tzqFvKkQkDvNbw4mDxYWY3/view?usp=sharing'
id = '1eUxIBU4oq5tzqFvKkQkDvNbw4mDxYWY3'
downloaded = drive.CreateFile({'id':id})
downloaded.GetContentFile('en_data.csv')

# To load raw data
# p_df = pd.read_csv('mbsa.csv')
# To load pre-processed data
en_data = pd.read_csv('en_data.csv',encoding='latin-1')
en_data = en_data.dropna()
en_data = en_data[en_data['Sentiment']!='Neutral']

print(len(en_data))
en_data.head()

# Reduce size for memory saving
en_data = en_data[en_data['Date']>'2019-06']

! pip install whatthelang
! pip install swifter

import numpy as np # linear algebra
import pandas as pd
import os
from whatthelang import WhatTheLang
import swifter
from google.colab import drive

from sklearn.model_selection import train_test_split
X = en_data.drop(columns =['Sentiment','Date','lang'],inplace=False)
y = en_data['Sentiment']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y)

X_train = X_train['text']
X_val = X_val['text']
y_train = y_train.replace({'Negative':0,'Positive':4})
y_val = y_val.replace({'Negative':0,'Positive':4})

# Drop en_data to save memory
del en_data

import sys
def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                         key= lambda x: -x[1])[:10]:
    print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))


import nltk
nltk.download('vader_lexicon')

# Rule based sentiment analysis with Vader
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# initialize our rules-based model
sid = SentimentIntensityAnalyzer()
for txt, lbl in zip(X_train[0:10], y_train[0:10]):
  print(f"{txt}\n{lbl}")
  # calculate the sentiment
  ss = sid.polarity_scores(txt)
  for k in sorted(ss):
    print('{0}: {1}, '.format(k, ss[k]), end='')
  print('\n------------------\n\n')

  # accuracy evaluation for VADER - rules-based
test_preds = []
n = len(X_train)
i = 0
for t in X_train:
  i += 1
  if i % 50000 == 0:
    print(f'{round(i/n*100)}%')
  ss = sid.polarity_scores(t)
  score = ss['compound']
  if score > 0:
    test_preds.append(4)  # positive
  else:
    test_preds.append(0)  # negative

print('test accuracy: {}'.format(np.sum(test_preds == y_train) / len(y_train)))
print('-------')

# print some of them out
for i in range(0, 325, 25):
  ss = sid.polarity_scores(X_train.iloc[i])
  print(X_train.iloc[i])
  for k in sorted(ss):
    print('{0}: {1}, '.format(k, ss[k]), end='')
  print('prediction: {}, true: {}'.format(test_preds[i], y_train.iloc[i]))
  print('------------------')

  # learning-based model in scikit-learn
  from sklearn.feature_extraction.text import CountVectorizer
  from sklearn.naive_bayes import MultinomialNB
  from sklearn.preprocessing import LabelEncoder

  # the input for our classifier is a term-frequency matrix
  vect = CountVectorizer()
  X = vect.fit_transform(X_train)

  # we also want to encode our labels so that they can be represented as integers
  label_enc = LabelEncoder()
  Y = label_enc.fit_transform(y_train)
  print(label_enc.classes_)


  # initialize the model
  nb = MultinomialNB()


  # training the model (this function does all of the heavy lifting)
  nb.fit(X, Y)
