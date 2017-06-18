import numpy as np
import pandas as pd

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn import svm
from sklearn.model_selection import cross_val_score


def import_data(filename):
	return pd.read_csv(filename, header=0).fillna('').values

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

def divide_train_test(data):
	train = data[:1610]
	test = data[1611:]
	return train, test

def train(data, labels):
	print(labels.shape)
	return svm.SVC(kernel='linear', C=1).fit(data, labels)

def cross_validate(data, labels, clsfr):
	return cross_val_score(clsfr, data, labels, cv=5)

def main():
	data = import_data("./../data/Combined_News_DJIA.csv")
	pre_processed = pre_process(data)
	sentiment_included = add_avg_sentiment(pre_processed)

	train_set, test_set = divide_train_test(sentiment_included)
	train_labels = train_set[:,1].ravel()
	train_sentiments = train_set[:,27].reshape(len(train_set), 1)
	test_labels = test_set[:,1].ravel()
	test_sentiments = test_set[:,27].reshape(len(test_set),1)

	clsfr = train(train_sentiments, train_labels.astype('int'))
	print("Training done.")
	results = cross_validate(test_sentiments, test_labels.astype('int'), clsfr)
	print(test_labels.mean())
	print(results)
	print("Avg = " + str(results.mean()))
	print("Cross validation done.")
	print("Done.")

if __name__ == "__main__":
	main()