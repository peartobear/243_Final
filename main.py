from flask import Flask, render_template, request
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer


app = Flask(__name__)

#loading the transfer learning model and the transformers

model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model.load_state_dict(torch.load("model.pt", map_location=torch.device('cpu')))
model.eval()

# function to generate summary


def generate_summary(article):
    max_summary_len = 500
    input_text = "summarize: " + article.strip()
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    summary_ids = model.generate(
        input_ids=input_ids,
        num_return_sequences=1,
        max_length=max_summary_len,
        num_beams=4,
        no_repeat_ngram_size=3,
        temperature=0.7,
        length_penalty=2.0,
        early_stopping=True,
    )
    summary = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)
    words = summary.split()
    k = []
    for i in words:
        if summary.count(i) >= 1 and i not in k:
            if i.endswith('s') and i[:-1] in k:
                continue
            k.append(i)
    summary = (' '.join(k))
  
    return summary

# Initializing the sentiment analyzer

sia = SentimentIntensityAnalyzer()

# function to analyze sentiment

def analyze_sentiment(summary):
    sentiment_score = sia.polarity_scores(summary)
    return sentiment_score

# function to render the index.html page

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        article = request.form['article']
        summary = generate_summary(article)
        sentiment_score_summary = analyze_sentiment(summary)
        sentiment_score_article = analyze_sentiment(article)
        return render_template('index.html', summary=summary, sentiment_score_summary=sentiment_score_summary, sentiment_score_article=sentiment_score_article)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)