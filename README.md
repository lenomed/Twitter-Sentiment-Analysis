# Twitter Sentiment Analysis

A Natural Language Processing (NLP) and machine learning-based Twitter sentiment analysis system that classifies tweets into **positive** or **negative** sentiments using Logistic Regression and TF-IDF vectorization.

## 📋 Overview

This project preprocesses raw Twitter data and trains a machine learning model to automatically classify tweet sentiment. It uses text stemming, stopword removal, and TF-IDF feature extraction to achieve accurate sentiment predictions.

**Key Features:**

- Binary classification (Positive/Negative sentiment)
- Text preprocessing with Porter Stemmer
- TF-IDF vectorization for feature extraction
- Logistic Regression classifier
- Model persistence with pickle serialization
- Comprehensive model evaluation metrics

---

## 📁 Project Structure

```
Twitter-Sentiment-Analysis/
├── app.py                      # Model training script
├── main.py                     # Inference/prediction script
├── sentiment_model.sav         # Trained Logistic Regression model
├── vectorizer.sav              # Fitted TF-IDF vectorizer
├── twitter_data.csv            # Dataset
├── requirements.txt            # Project dependencies
├── .gitignore
└── README.md
```

---

## 🚀 Installation

### Prerequisites

- Python 3.7+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/lenomed/Twitter-Sentiment-Analysis.git
cd Twitter-Sentiment-Analysis

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Required Libraries

```
pandas
numpy
matplotlib
seaborn
nltk
scikit-learn
```

---

## 💻 Usage

### 1. Training the Model

Run `app.py` to train the sentiment classifier on your dataset:

```bash
python app.py
```

**What it does:**

- Loads `twitter_data.csv` dataset
- Performs data exploration and visualization
- Applies text preprocessing (stemming, stopword removal)
- Converts text to TF-IDF vectors
- Trains a Logistic Regression model
- Saves the trained model and vectorizer

**Output:**

- `sentiment_model.sav` - Trained model
- `vectorizer.sav` - Fitted vectorizer
- Classification report, confusion matrix, and accuracy score printed to console

### 2. Making Predictions

Run `main.py` to classify new tweets:

```bash
python main.py
```

**What it does:**

- Loads the pre-trained model and vectorizer
- Preprocesses test tweets using the same stemming logic
- Generates sentiment predictions
- Displays results in a readable format

**Example Output:**

```
                                                text   sentiment
0  I absolutely love this phone. The battery l...    Positive
1  This is the worst purchase I have ever made.    Negative
2        The service was okay, nothing special.    Negative
3              I'm very happy with the results.    Positive
4         I regret spending money on this product.    Negative
```

---

## 🔧 Data Preprocessing Pipeline

The project applies the following text preprocessing steps:

```
Raw Tweet Text
    ↓
1. Remove special characters & numbers (keep only a-zA-Z)
    ↓
2. Convert to lowercase
    ↓
3. Tokenize (split into words)
    ↓
4. Remove stopwords (common words like 'the', 'is', etc.)
    ↓
5. Porter Stemming (reduce words to root form)
    ↓
Cleaned & Stemmed Text
```

**Example:**

```python
Input:  "I absolutely LOVE this phone!!! It's amazing @twitter"
Output: "absolutli love phone amaz"
```

---

## 🧠 Model Details

### Feature Extraction

- **TF-IDF Vectorizer**: Converts text to numerical features
- **Vocabulary**: Automatically built from training data

### Classification Model

- **Algorithm**: Logistic Regression
- **Hyperparameters**: `max_iter=1000`
- **Input**: TF-IDF vectors
- **Output**: Binary classification (0=Negative, 1=Positive)

### Dataset Information

- **Columns**: target, Id, date, flag, user, text
- **Target Variable**:
  - 4 → 1 (Positive sentiment)
  - 0 → 0 (Negative sentiment)
- **Train-Test Split**: 60% train, 40% test
- **Split Strategy**: Stratified (maintains class distribution)

---

## 📊 Model Evaluation

The trained model is evaluated using:

1. **Accuracy Score**: Overall correctness of predictions
2. **Classification Report**: Precision, recall, F1-score per class
3. **Confusion Matrix**: True positives, true negatives, false positives, false negatives

**Run after training to see:**

```
              precision    recall  f1-score   support

           0       0.XX      0.XX      0.XX       XXX
           1       0.XX      0.XX      0.XX       XXX

    accuracy                           0.XX       XXX
   macro avg       0.XX      0.XX      0.XX       XXX
weighted avg       0.XX      0.XX      0.XX       XXX
```

---

## 📝 Key Code Sections

### Text Stemming Function

```python
def stemming(content):
    stemmed = re.sub('[^a-zA-Z]', ' ', content)  # Remove non-letters
    stemmed = stemmed.lower()                     # Lowercase
    stemmed = stemmed.split()                     # Tokenize
    stemmed = [p_stem.stem(word) for word in stemmed
               if not word in stopwords.words('english')]  # Stem & remove stopwords
    stemmed = ' '.join(stemmed)
    return stemmed
```

### Model Training

```python
# Vectorize training data
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save for later use
pickle.dump(model, open('sentiment_model.sav', 'wb'))
pickle.dump(vectorizer, open('vectorizer.sav', 'wb'))
```

### Making Predictions

```python
# Load saved model & vectorizer
model = pickle.load(open('sentiment_model.sav', 'rb'))
vectorizer = pickle.load(open('vectorizer.sav', 'rb'))

# Preprocess new text
cleaned_text = stemming(new_tweet)

# Vectorize & predict
X_vectorized = vectorizer.transform([cleaned_text])
prediction = model.predict(X_vectorized)

# Map to sentiment labels
sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
```

---

## 📦 Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

**Project Requirements:**

- `numpy` - Numerical computing
- `pandas` - Data manipulation and analysis
- `scikit-learn` - Machine Learning models and evaluation
- `matplotlib` - Data visualization
- `seaborn` - Statistical data visualization
- `nltk` - Natural Language Processing and text preprocessing
- `textblob` - Simplified NLP (alternative sentiment analysis)
- `tweepy` - Twitter API integration (for fetching tweets)
- `wordcloud` - Word cloud visualization

**Current Usage:**
Currently, the main scripts use: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, and `nltk`.

**Optional Libraries:**
`textblob`, `tweepy`, and `wordcloud` are available for extending the project with features like:

- Real-time tweet collection via Twitter API
- Alternative sentiment analysis methods
- Sentiment distribution visualizations

---

## 🔄 Workflow

```
1. Data Loading & Exploration (app.py)
         ↓
2. Text Preprocessing & Cleaning
         ↓
3. Feature Extraction (TF-IDF)
         ↓
4. Train-Test Split
         ↓
5. Model Training (Logistic Regression)
         ↓
6. Model Evaluation & Metrics
         ↓
7. Save Model & Vectorizer
         ↓
8. Load for Predictions (main.py)
         ↓
9. Preprocess New Data
         ↓
10. Generate Predictions
         ↓
11. Display Results
```

---

## 📈 Performance Metrics

After training, the model outputs:

- **Accuracy**: Overall correctness percentage
- **Precision**: Correctness of positive predictions
- **Recall**: Coverage of actual positive instances
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Breakdown of correct/incorrect predictions

---

## 📄 requirements.txt

Your `requirements.txt` file should contain:

```
numpy
pandas
scikit-learn
matplotlib
seaborn
nltk
textblob
tweepy
wordcloud
```

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## 🛠️ Customization

### Change Train-Test Split

In `app.py`, modify:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=101
)
```

Adjust `test_size` (e.g., 0.3 for 70-30 split)

### Add New Test Data

In `main.py`, modify the test data list:

```python
test_data = [
    "Your new tweet here",
    "Another tweet to analyze",
]
```

### Adjust Model Parameters

In `app.py`, modify:

```python
model = LogisticRegression(max_iter=1000)
```

Try different values or algorithms (e.g., SVM, Random Forest)

---

## 📂 Required Files

For the project to run, you need:

- `twitter_data.csv` - Training dataset with columns: target, Id, date, flag, user, text
- `requirements.txt` - Python dependencies

---

## 🤝 Contributing

Pull requests are welcome! Feel free to:

- Improve model accuracy
- Add new features
- Optimize preprocessing
- Enhance documentation

---

---

## 👨‍💻 Author

**Lenomed** - [GitHub](https://github.com/lenomed)

---

## 📞 Support

For issues or questions, please open a GitHub issue in the repository.
