import numpy as np
import pandas as pd

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn import svm
from sklearn.model_selection import cross_val_score

from dateutil.parser import parse
from datetime import timedelta

import random
import re


'''

This script tries to predict today's, tommorow's and the day after tomorrow's 
IBM stock with headlines on technology in which the term 'ibm' resides.

'''

##### FLAGS
train_on_avg_sentiments = False


def import_data(filename):
	return pd.read_csv(filename, header=0, sep = '\t').fillna('').values


def pre_process(data):
	return data


def add_sentiment(data):
	sid = SentimentIntensityAnalyzer()
	avgs = np.empty((len(data),1))
	for i in range(0,len(data)):
		field = data[i,12]
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


def filter_data(data):
	output = np.empty((1,4))
	for line in data:
		ms = line[9]
		msTom = line[10]
		msDayAfterTom = line[11]
		sentiment = line[13]

		output = np.append(output, [[ms, msTom, msDayAfterTom, sentiment]], axis=0)
	return output[1:]


def divide_train_test(data):
	random.seed(2)
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
	data = import_data("./../data/all_data/combined_IBM_IBM_tech_news.csv")
	pre_processed = pre_process(data)
	sentiment_included = add_sentiment(pre_processed)
	if train_on_avg_sentiments:
		avg_sentiments = calc_avg_sentiments(sentiment_included)
	else:
		avg_sentiments = filter_data(sentiment_included)

	print("Data exists of " + str(avg_sentiments.shape[0]) + " cases.\n")

	# Predict today's stock
	train_set, test_set = divide_train_test(avg_sentiments)
	train_labels = train_set[:,0].ravel()
	train_sentiments = train_set[:,3].reshape(len(train_set), 1)
	test_labels = test_set[:,0].ravel()
	test_sentiments = test_set[:,3].reshape(len(test_set),1)

	clsfr = train(train_sentiments, train_labels.astype('int'))
	print("Training done.")
	print("Pedicting today's stock...")
	scores = cross_validate(test_sentiments, test_labels.astype('int'), clsfr)
	print(scores)
	print("Average: " + str(scores.mean()))
	print("Cross validation done.\n")

	# Predict tomorrow's stock
	train_set, test_set = divide_train_test(avg_sentiments)
	train_labels = train_set[:,1].ravel()
	train_sentiments = train_set[:,3].reshape(len(train_set), 1)
	test_labels = test_set[:,1].ravel()
	test_sentiments = test_set[:,3].reshape(len(test_set),1)

	clsfr = train(train_sentiments, train_labels.astype('int'))
	print("Training done.")
	print("Predicting tomorrow's stock...")
	scores = cross_validate(test_sentiments, test_labels.astype('int'), clsfr)
	print(scores)
	print("Average: " + str(scores.mean()))
	print("Cross validation done.\n")

	# Predict the day after tomorrow's stock
	train_set, test_set = divide_train_test(avg_sentiments)
	train_labels = train_set[:,2].ravel()
	train_sentiments = train_set[:,3].reshape(len(train_set), 1)
	test_labels = test_set[:,2].ravel()
	test_sentiments = test_set[:,3].reshape(len(test_set),1)

	clsfr = train(train_sentiments, train_labels.astype('int'))
	print("Training done.")
	print("Predicting the day after tomorrow's stock...")
	scores = cross_validate(test_sentiments, test_labels.astype('int'), clsfr)
	print(scores)
	print("Average: " + str(scores.mean()))
	print("Cross validation done.")
	print("Done.")


if __name__ == "__main__":
	main()