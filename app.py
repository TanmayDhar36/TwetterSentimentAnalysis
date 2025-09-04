from flask import Flask, render_template, request
import pickle
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Initialize Flask app
app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open('D:/Python Practice/TweeterSentimentAnaysis/trained_model.sav', 'rb'))
vectorizer = pickle.load(open('D:/Python Practice/TweeterSentimentAnaysis/vectorizer.pkl', 'rb'))

# Preprocessing function (same as training)
port_stem = PorterStemmer()
def stremming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    return ' '.join(stemmed_content)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        tweet = request.form['tweet']
        cleaned_tweet = stremming(tweet)
        vectorized_tweet = vectorizer.transform([cleaned_tweet])
        prediction = model.predict(vectorized_tweet)[0]

        sentiment = "Positive" if prediction == 1 else "Negative"
        return render_template('result.html', tweet=tweet, sentiment=sentiment)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
