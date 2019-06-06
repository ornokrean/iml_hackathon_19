from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from task2.src.data_processor import prepare_data
import numpy as np
from matplotlib import pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

CSV_PATH = "Crimes_since_2005.csv"
# CSV_PATH = "partial_data"
CLASS_HEADER = 'Primary Type'


def split_data(data, ratio):
	labels = data.pop(CLASS_HEADER)
	split_result = train_test_split(data, labels, test_size=ratio)
	data[CLASS_HEADER] = labels
	return split_result


def get_success_rate(learner, test, test_labels):
	prediction_fails = learner.predict(test) == test_labels
	ltest = list(prediction_fails)
	return round(ltest.count(True) / float(len(ltest)), 3)


def get_succ_rate_for_params(data, split_ratio, tree_depth):
	# print("Testing learner for params:")
	# print("\tsplit ratio:",split_ratio)
	# print("\ttree depth:",tree_depth)
	train, test, train_labels, test_labels = split_data(data, split_ratio)
	# print("\ttrain data:", train.shape, "train_labels:", train_labels.shape)
	# print("\ttest data:", test.shape, "test_labels:", test_labels.shape)
	learner = DecisionTreeClassifier(max_depth=tree_depth).fit(train, train_labels)
	succ_rate = get_success_rate(learner, test, test_labels)
	# print("Got success rate:",succ_rate)
	return succ_rate


def get_tree_success_rate(data, ratio, iterations,depth = 6):

	mean = 0
	for i in range(iterations):
		mean+= get_succ_rate_for_params(data, ratio, depth)
	return mean/iterations




def main():
	# get processed data
	data = prepare_data(CSV_PATH, 15000)



	# get average error rate for a single tree
	print("Single tree success rate: ",get_tree_success_rate(data,0.3,5))





	# fit on training data notes

	# todo this learner yields ~0.3 success rate
	# logistic_regression_learner = LogisticRegression(solver = "lbfgs",multi_class="multinomial",max_iter = 5000).fit(train,train_labels)

	# todo this learner yields 0.5-0.6 success rate
	# dtree_model = DecisionTreeClassifier(max_depth=7).fit(train, train_labels)



if __name__ == '__main__':
	main()