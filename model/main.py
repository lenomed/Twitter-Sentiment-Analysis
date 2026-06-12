import pickle
import pandas as pd
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Load the trained model and vectorizer
model = pickle.load(open('sentiment_model.sav', 'rb'))
vectorizer = pickle.load(open('vectorizer.sav', 'rb'))  # You need to save this too

# i am creating a test set here
test_data = [
    "I absolutely love this phone. The battery life is amazing.",
    "This is the worst purchase I have ever made.",
    "The service was okay, nothing special.",
    "I'm very happy with the results.",
    "I regret spending money on this product."
]

col_names = ['text']
df = pd.DataFrame(test_data, columns=col_names)
print(df.head())

# Preprocess test data (same as training)
p_stem = PorterStemmer()

def stemming(content):
    stemmed = re.sub('[^a-zA-Z]', ' ', content) 
    stemmed = stemmed.lower()   
    stemmed = stemmed.split()
    stemmed = [p_stem.stem(word) for word in stemmed if not word in stopwords.words('english')]
    stemmed = ' '.join(stemmed)
    return stemmed

df['stemmed_text'] = df['text'].apply(stemming)

# Vectorize using the same vectorizer from training
X_test_vectorized = vectorizer.transform(df['stemmed_text'])

# Make predictions
predictions = model.predict(X_test_vectorized)

# Display results
df['sentiment'] = predictions
df['sentiment'] = df['sentiment'].map({1: 'Positive', 0: 'Negative'})
print(df[['text', 'sentiment']])