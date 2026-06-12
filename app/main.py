import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



# Load the dataset
data = pd.read_csv(r'C:\\Users\\kelec\\lenomed\\Twitter-Sentiment-Analysis\\twitter_data.csv', encoding='cp1252')

# checking out the characteristics of the data set
print(data.head(10))
print(data.info())
print(data.describe())
print(data.columns)

#downloading stop words
#nltk.download('stopwords')

#print(stopwords.words('english'))

#rename the columns

column_names = ['target','target_Id', 'Id', 'data', 'user', 'text']
data = pd.read_csv(r'C:\\Users\\kelec\\lenomed\\Twitter-Sentiment-Analysis\\twitter_data.csv',names=column_names, encoding='cp1252')

print(data.head(10))

#print(f'number of missing values: {data.isnull().sum()}')
#sns.heatmap(data.isnull())
#plt.show()

# checking the distribution at target column
target_categorical_values = data['target'].value_counts()
print(target_categorical_values)
sns.countplot(x='target', data=data)
plt.show()

data = data.map({'4':1})