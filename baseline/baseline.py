import numpy as np
import pandas as pd

from nltk.sentiment.vader import SentimentIntensityAnalyzer


def import_data(filename):
	return pd.read_csv(filename, header=0).fillna('').as_matrix()

def pre_process(data):
	for row in data[0:476]:
		for field in row[2:]:
			if not field == '':
				field = field[1:]	# Remove first 'b'
	return data

def add_avg_sentiment(data):
	sid = SentimentIntensityAnalyzer()
	avgs = np.empty((len(data),1))
	for i in range(0,len(data)):
		sentiments = []
		for field in data[i][2:]:
			sentiments.append(sid.polarity_scores(field)['compound'])
		avg = float(sum(sentiments))/len(sentiments)
		avgs[i] = avg
	return np.append(data, avgs, axis=1)

def main():
	data = import_data("./../data/Combined_News_DJIA.csv")
	print(data.shape)
	pre_processed = pre_process(data)
	sentiment_included = add_avg_sentiment(pre_processed)

if __name__ == "__main__":
	main()