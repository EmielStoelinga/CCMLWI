import numpy as np
import pandas as pd

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn import svm
from sklearn.model_selection import cross_val_score

from dateutil.parser import parse
from datetime import timedelta

import random


def import_data(filename):
	return pd.read_csv(filename, header=0, sep = '\t').fillna('').values

def pre_process(data):
	return data

def add_sentiment(data):
	sid = SentimentIntensityAnalyzer()
	avgs = np.empty((len(data),1))
	for i in range(0,len(data)):
		field = data[i,11]
		avgs[i] = sid.polarity_scores(field)['compound']
	return np.append(data, avgs, axis=1)

def calc_avg_sentiments(data):
	previousDate = parse(data[0,8]) - timedelta(days=1)
	previousMs = -1
	previousMsTom = -1
	tempSents = [0]
	output = np.empty((1,3))
	for line in data:
		date = parse(line[8])
		sentiment = line[12]
		ms = line[9]
		msTom = line[10]
		if date > previousDate:
			avgSent = sum(tempSents)/len(tempSents)
			output = np.append(output,[[ms, msTom, avgSent]], axis=0)

			tempSents = []
			tempSents.append(sentiment)
			previousDate = date
			previousMs = ms
			previousMsTom = msTom
		elif date == previousDate:
			tempSents.append(sentiment)
		else:
			print("Wrong date order.")
	return output[1:]	# First row was to initialize. Ugly, but works.

def divide_train_test(data):
	random.seed(1)
	random.shuffle(data)

	split = .7 * len(data)
	train = data[:int(split)]
	test = data[int(split):]
	return train, test

def train(data, labels):
	return svm.SVC(kernel='linear', C=1).fit(data, labels)

def cross_validate(data, labels, clsfr):
	return cross_val_score(clsfr, data, labels, cv=5)

def main():
	data = import_data("./../data/combined_technology_news_stocks.csv")
	pre_processed = pre_process(data)
	sentiment_included = add_sentiment(pre_processed)
	avg_sentiments = calc_avg_sentiments(sentiment_included)

	# Predict today's stock
	train_set, test_set = divide_train_test(avg_sentiments)
	train_labels = train_set[:,0].ravel()
	train_sentiments = train_set[:,2].reshape(len(train_set), 1)
	test_labels = test_set[:,0].ravel()
	test_sentiments = test_set[:,2].reshape(len(test_set),1)

	clsfr = train(train_sentiments, train_labels.astype('int'))
	print("Training done.")
	print("Pedicting today's stock...")
	print(cross_validate(test_sentiments, test_labels.astype('int'), clsfr))
	print("Cross validation done.")
	print("Done.")

	# Predict tomorrow's stock
	train_set, test_set = divide_train_test(avg_sentiments)
	train_labels = train_set[:,1].ravel()
	train_sentiments = train_set[:,2].reshape(len(train_set), 1)
	test_labels = test_set[:,1].ravel()
	test_sentiments = test_set[:,2].reshape(len(test_set),1)

	clsfr = train(train_sentiments, train_labels.astype('int'))
	print("Training done.")
	print("Predicting tomorrow's stock...")
	print(cross_validate(test_sentiments, test_labels.astype('int'), clsfr))
	print("Cross validation done.")
	print("Done.")

if __name__ == "__main__":
	main()