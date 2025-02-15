from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


# here we do stemming and lemmatization bc we want to reduce the words to their root form => only keep the important words from tweet
def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = text.split()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        tweet = request.form['tweet']
        # get the tweet and preprocess it (remove links, @, special characters, lowercase, remove stopwords, lemmatize)
        processed_tweet = preprocess_text(tweet)
        # vectorize the tweet (convert it to a vector of numbers as the model can only work with numbers)
        tweet_vector = vectorizer.transform([processed_tweet])
        # make prediction
        prediction = model.predict(tweet_vector)[0]
        # get the probability of the prediction
        probability = model.predict_proba(tweet_vector)[0].max()
        
        sentiment = "Positive" if prediction == 1 else "Negative" 
        return render_template('result.html', tweet=tweet, sentiment=sentiment, confidence=round(probability*100, 2))

if __name__ == '__main__':
    app.run(debug=True)