from flask import Flask, request, render_template
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from string import punctuation
import re
from nltk.corpus import stopwords

# Download NLTK stopwords if not present
nltk.download('stopwords')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def analyze_sentiment():
    result = None
    text_input = ""
    scores = {}
    sentiment_label = ""

    if request.method == 'POST':
        stop_words = set(stopwords.words('english'))

        # Step 1: Get text input and clean it
        text_input = request.form['text1']
        text_clean = text_input.lower()
        text_clean = re.sub(r'\d+', '', text_clean)  # remove digits
        text_clean = ''.join(c for c in text_clean if c not in punctuation)
        text_clean = ' '.join(word for word in text_clean.split() if word not in stop_words)

        # Step 2: Perform sentiment analysis
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text_clean)
        compound = scores['compound']

        # Normalize compound to a 0–1 range if needed
        normalized = round((1 + compound) / 2, 2)

        # Step 3: Determine sentiment label
        if compound >= 0.05:
            sentiment_label = "Positive"
        elif compound <= -0.05:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"

        result = {
            'final': normalized,
            'pos': scores['pos'],
            'neu': scores['neu'],
            'neg': scores['neg'],
            'compound': compound,
            'sentiment_label': sentiment_label
        }

    return render_template('form.html', text_input=text_input, result=result)

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5002, threaded=True)
