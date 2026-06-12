import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
import pickle

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



# Load the dataset
data = pd.read_csv(r'C:\Users\kelec\source\repos\Twitter-Sentiment-Analysis\twitter_data.csv', encoding='cp1252')

# checking out the characteristics of the data set
print(data.head(10))
print(data.info())
print(data.describe())
print(data.columns)

#downloading stop words
#nltk.download('stopwords')

#print(stopwords.words('english'))

#rename the columns

column_names = ['target', 'Id', 'date','flag', 'user', 'text']
data = pd.read_csv(r'C:\Users\kelec\source\repos\Twitter-Sentiment-Analysis\twitter_data.csv',names=column_names, encoding='cp1252')

print(data.head(10))

#print(f'number of missing values: {data.isnull().sum()}')
#sns.heatmap(data.isnull())
#plt.show()

# checking the distribution at target column
target_categorical_values = data['target'].value_counts()
print(target_categorical_values)
sns.countplot(x='target', data=data)
#plt.show()

data['target'] = data['target'].map({4: 1, 0: 0}).astype(int)
print(data['target'])

# stemming
# it is the process of reeducing a word to it's root word
# i am doing all these to resuce the data set as much as possible and to reduce ambiguity and processing time.

p_stem = PorterStemmer()

print(data['text'])


def stemming(content):

    stemmed = re.sub('[^a-zA-Z]', ' ', content) 
    stemmed = stemmed.lower()   
    stemmed= stemmed.split()
    stemmed=[p_stem.stem(word) for word in stemmed if not word in stopwords.words('english')]
    stemmed = ' '.join(stemmed)
    return stemmed

data['stemmed_output']= data['text'].apply(stemming)
print(f'this is stem output{data['stemmed_output']}')

data.drop(['Id', 'date', 'flag', 'user', 'text'],axis=1, inplace=True)

print(data.head(20))

# spliting my data set

X = data.drop('target', axis=1).values
y= data['target'].values

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=.4, stratify=y, random_state=101)

#converting texts to vectors using Tfidf Vectorizer
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

print(f'Vectorized X_train: {X_train}')
print(f'Vectorized X_test: {X_test}')

#instantiate the model

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

# Model Evaluation
print(classification_report(y_test,  predictions))
print(confusion_matrix(y_test,  predictions))
print(accuracy_score(y_test,  predictions))

# saving the trained model
file = 'sentiment_model.sav'
pickle.dump(model, open(file, 'wb'))

# saving the vectorizer
vectorizer_file = 'vectorizer.sav'
pickle.dump(vectorizer, open(vectorizer_file, 'wb'))

