import pandas as pd
import numpy as np
from sklearn.feature_extraction.text  import TfidfVectorizer

# tfidf - term frequency - inverse document frequency

# tf - how many times the word appeared in the document
# idf - how rare the word is will have the high score and vice versa

news =  pd.read_csv('./fake_news_dataset.csv')

# print(news.isnull().sum())
 # we see many empty values so lets fill with empty strings
new_news=news.fillna('')

# print(new_news.isnull().sum())
 # lets combine the data columns of autor and title  in to new_news.Content
new_news['Content'] = new_news['author']+' '+ new_news ['title']
# we now save this data in var called X
X = (new_news['Content'])

# we load the tfidf module in load variable
load = TfidfVectorizer()
# and we fix the data of new_news.content
load.fit(X)
# Transform and convert the text to values
Y = load.transform(X)

print(Y)