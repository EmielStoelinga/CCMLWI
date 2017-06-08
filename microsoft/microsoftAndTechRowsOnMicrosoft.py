import numpy as np
import pandas as pd

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn import svm
from sklearn.model_selection import cross_val_score

from dateutil.parser import parse
from datetime import timedelta

import random
import re

from tqdm import tqdm

'''

This script tries to predict today's, tommorow's and the day after tomorrow's Microsoft stock with headlines on technology in which the term 'microsoft' resides.

'''


##### FLAGS
train_on_avg_sentiments = False


def import_data(filename):
	return pd.read_csv(filename, header=0, sep = '\t').fillna('')


def filter_out_microsoft(data):
	'''
	output = np.empty((1,data.shape[1]))
	for i in tqdm(range(0, len(data))):
		if not re.search('microsoft', data[i, 12]):
			output = np.append(output, np.reshape(data[i], (1, data.shape[1])), axis=0)
	return output[1:].values
	'''
	return data[~data['normalized_headline'].str.contains("microsoft")].values


def add_sentiment(data):
	sid = SentimentIntensityAnalyzer()
	avgs = np.empty((len(data),1))
	for i in tqdm(range(0,len(data))):
		field = data[i,12]
		avgs[i] = sid.polarity_scores(field)['compound']
	return np.append(data, avgs, axis=1)


def calc_avg_sentiments(data):
	previousDate = parse(data[0,8]) - timedelta(days=1)
	tempSents = [0]
	output = np.empty((1,2))
	for line in data:
		date = parse(line[8])
		sentiment = line[13]
		if date > previousDate:
			avgSent = sum(tempSents)/len(tempSents)
			output = np.append(output, [[previousDate, avgSent]], axis=0)

			tempSents = []
			tempSents.append(sentiment)
			previousDate = date
		elif date == previousDate:
			tempSents.append(sentiment)
		else:
			print("Wrong date order during calculating average.")
	avgSent = sum(tempSents)/len(tempSents)
	output = np.append(output, [[previousDate, avgSent]], axis=0)
	
	return output[2:]	# First row was to initialize. Ugly, but works.


def combine_data(microsoft_data, tech_data):
	tech_index = 0
	avg_sents = np.empty((len(microsoft_data), 1))
	for i in tqdm(range(0, len(microsoft_data))):
		microsoft_date = parse(microsoft_data[i, 8])
		if microsoft_date == tech_data[tech_index, 0]:
			avg_sents[i] = tech_data[tech_index, 1]
		else:
			if not microsoft_date == tech_data[tech_index+1, 0]:
				print "Day of Microsoft headlines missing."
				tech_index += 1
			else:
				tech_index += 1
				avg_sents[i] = tech_data[tech_index, 1]
	microsoft_data = np.append(microsoft_data, avg_sents, axis=1)

	return microsoft_data


def filter_data(data):
	output = np.empty((1,5))
	for line in data:
		ms = line[9]
		msTom = line[10]
		msDayAfterTom = line[11]
		sentiment = line[13]
		sentiment_tech = line[14]

		output = np.append(output, [[ms, msTom, msDayAfterTom, sentiment, sentiment_tech]], axis=0)
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
	microsoft_data = import_data("./../data/all_data/combined_MSFT_microsoft_tech_news.csv").values
	tech_data = import_data("./../data/all_data/combined_MSFT_tech_news.csv")
	print("Data reading done.")
	filtered_tech_data = filter_out_microsoft(tech_data)
	print("Filtering tech data done.")

	microsoft_sentiment_included = add_sentiment(microsoft_data)
	print("Adding sentiments to microsoft done.")
	tech_sentiment_included = add_sentiment(filtered_tech_data)
	print("Adding sentiments to technology done.")

	avg_tech_sentiments = calc_avg_sentiments(tech_sentiment_included)
	print("Calculating average sentiments done.")

	combined_data = combine_data(microsoft_sentiment_included, avg_tech_sentiments)
	print("Combining data done.")

	filtered_data = filter_data(combined_data)
	print("Data filtered.")

	print("Data exists of " + str(filtered_data.shape[0]) + " cases.\n")

	# Predict today's stock
	train_set, test_set = divide_train_test(filtered_data)
	
	train_labels = train_set[:,0].ravel()
	train_msft_sentiments = train_set[:,3].reshape(len(train_set), 1)
	train_tech_sentiments = train_set[:,4].reshape(len(train_set), 1)
	train_sentiments = np.append(train_msft_sentiments, train_tech_sentiments, axis = 1)

	test_labels = test_set[:,0].ravel()
	test_msft_sentiments = test_set[:,3].reshape(len(test_set),1)
	test_tech_sentiments = test_set[:,4].reshape(len(test_set),1)
	test_sentiments = np.append(test_msft_sentiments, test_tech_sentiments, axis = 1)

	clsfr = train(train_sentiments, train_labels.astype('int'))
	print("Training done.")
	print("Pedicting today's stock...")
	scores = cross_validate(test_sentiments, test_labels.astype('int'), clsfr)
	print(scores)
	print("Average: " + str(scores.mean()))
	print("Cross validation done.\n")

	# Predict tomorrow's stock
	train_set, test_set = divide_train_test(filtered_data)

	train_labels = train_set[:,1].ravel()
	train_msft_sentiments = train_set[:,3].reshape(len(train_set), 1)
	train_tech_sentiments = train_set[:,4].reshape(len(train_set), 1)
	train_sentiments = np.append(train_msft_sentiments, train_tech_sentiments, axis = 1)
	
	test_labels = test_set[:,1].ravel()
	test_msft_sentiments = test_set[:,3].reshape(len(test_set),1)
	test_tech_sentiments = test_set[:,4].reshape(len(test_set),1)
	test_sentiments = np.append(test_msft_sentiments, test_tech_sentiments, axis = 1)

	clsfr = train(train_sentiments, train_labels.astype('int'))
	print("Training done.")
	print("Predicting tomorrow's stock...")
	scores = cross_validate(test_sentiments, test_labels.astype('int'), clsfr)
	print(scores)
	print("Average: " + str(scores.mean()))
	print("Cross validation done.")
	print("Done.")

	# Predict the day after tomorrow's stock
	train_set, test_set = divide_train_test(filtered_data)
	
	train_labels = train_set[:,2].ravel()
	train_msft_sentiments = train_set[:,3].reshape(len(train_set), 1)
	train_tech_sentiments = train_set[:,4].reshape(len(train_set), 1)
	train_sentiments = np.append(train_msft_sentiments, train_tech_sentiments, axis = 1)

	test_labels = test_set[:,2].ravel()
	test_msft_sentiments = test_set[:,3].reshape(len(test_set),1)
	test_tech_sentiments = test_set[:,4].reshape(len(test_set),1)
	test_sentiments = np.append(test_msft_sentiments, test_tech_sentiments, axis = 1)

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