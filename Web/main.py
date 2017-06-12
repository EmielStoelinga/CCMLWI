from flask import Flask
from flask import render_template
from flask.ext.cache import Cache
from keras.models import load_model
import feedparser
import tensorflow as tf
import sys
sys.path.append("./LSTM")
from tf_data import TF_Data


microsoft_model = {"model": load_model("./LSTM/combined_MSFT_tech_news_day_after_tomorrow_.hdf5"), "data": TF_Data("./data/combined_MSFT_tech_news.csv", top_words=2000)}
google_model = {"model": load_model("./LSTM/combined_GOOG_tech_news_day_after_tomorrow_.hdf5"), "data": TF_Data("./data/combined_GOOG_tech_news.csv", top_words=2000)}
ibm_model = {"model": load_model("./LSTM/combined_IBM_tech_news_day_after_tomorrow_.hdf5"), "data": TF_Data("./data/combined_IBM_tech_news.csv", top_words=2000)}


app = Flask(__name__)
cache = Cache(app,config={'CACHE_TYPE': 'simple'})
@app.route("/")
@cache.cached(timeout=50)
def index():
    GOOG_feed = feedparser.parse("https://feeds.finance.yahoo.com/rss/2.0/headline?s=GOOG&region=US&lang=en-US")['entries']
    for f in GOOG_feed:
        sentence = google_model["data"].test_sentence(f['title'])        
        f['LSTM'] = google_model["model"].predict(sentence)[0][0]


    IBM_feed = feedparser.parse("https://feeds.finance.yahoo.com/rss/2.0/headline?s=IBM&region=US&lang=en-US")['entries']
    for f in IBM_feed:
        sentence = ibm_model["data"].test_sentence(f['title'])        
        f['LSTM'] = ibm_model["model"].predict(sentence)[0][0]
    MSFT_feed = feedparser.parse("https://feeds.finance.yahoo.com/rss/2.0/headline?s=MSFT&region=US&lang=en-US")['entries']
    for f in MSFT_feed:
        sentence = microsoft_model["data"].test_sentence(f['title'])        
        f['LSTM'] = microsoft_model["model"].predict(sentence)[0][0]



    return render_template("./index.html",feeds=[GOOG_feed,  IBM_feed, MSFT_feed])

if __name__ == "__main__":
    app.run()