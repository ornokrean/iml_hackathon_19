from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from task2.src.data_processor import prepare_data
import numpy as np
from matplotlib import pyplot as plt

from pandas import DataFrame
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
	prediction_success_indices = learner.predict(test) == test_labels
	ltest = list(prediction_success_indices)
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


def test_ada(data,split_ratio=0.3):

	train, test, train_labels, test_labels = split_data(data, split_ratio)

	ab = AdaBoostClassifier(
		DecisionTreeClassifier(max_depth=8),
		n_estimators=122,
		learning_rate=1.1)

	ab.fit(train, train_labels)
	return get_success_rate(ab,test,test_labels)

def test_KNN(data,split_ratio=0.3):
	train, test, train_labels, test_labels = split_data(data, split_ratio)

	knn=KNeighborsClassifier(n_neighbors = 8).fit(train, train_labels)

	knn.fit(train, train_labels)
	return get_success_rate(knn,test,test_labels)

def test_random_forest(data,split_ratio=0.3):
	train, test, train_labels, test_labels = split_data(data, split_ratio)

	rf = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=30)

	rf.fit(train, train_labels)

	return get_success_rate(rf,test,test_labels)

def main():

	# get processed data
	data = prepare_data(CSV_PATH, 10000)


	# todo this learner yields 0.3-0.4 success rate
	print("ada success:",test_ada(data))
	# todo this learner yields 0.3-0.4 success rate
	print("knn success:",test_KNN(data))
	# todo this learner yields 0.5-0.6 success rate
	print("Single tree success rate: ",get_tree_success_rate(data,0.3,5))
	# todo this learner yields 0.5-0.6 success rate
	print("Random forest success rate: ",test_random_forest(data,0.3))





	# fit on training data notes

	# todo this learner yields ~0.3 success rate
	# logistic_regression_learner = LogisticRegression(solver = "lbfgs",multi_class="multinomial",max_iter = 5000).fit(train,train_labels)

	# todo this learner yields 0.5-0.6 success rate
	# dtree_model = DecisionTreeClassifier(max_depth=7).fit(train, train_labels)



if __name__ == '__main__':
	main()