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


def get_err_rate_for_params(data,split_ratio,tree_depth):
	print("Testing learner for params:")
	print("\tsplit ratio:",split_ratio)
	print("\ttree depth:",tree_depth)
	train, test, train_labels, test_labels = split_data(data, 0.7)
	print("\ttrain data:", train.shape, "train_labels:", train_labels.shape)
	print("\ttest data:", test.shape, "test_labels:", test_labels.shape)
	learner = DecisionTreeClassifier(max_depth=tree_depth).fit(train, train_labels)
	succ_rate = get_success_rate(learner, test, test_labels)
	print("Got success rate:",succ_rate)
	return succ_rate





def main():
	# get processed data
	data = prepare_data(CSV_PATH, 150000)

	# split the data into training, validation and test
	# train, succ_rate, train_labels, test_labels = split_data(data,0.7)
	print("original data shape:", data.shape)

	# todo testing various parametrs on same learner
	for ratio in [i / 10.0 for i in range(1, 10)]:
		results = [0]
		for depth in range(1,10):
			err_rate = get_err_rate_for_params(data,ratio,depth)
			results.append(err_rate)
			# results+=[[ratio,depth,err_rate]]
		plt.title("success rate for tree with split ratio {}".format(ratio))
		plt.plot(results)
		plt.yticks([i/10.0 for i in range(11)])
		plt.show()

	# for result in results:
	# 	print("Ratio={}, Depth={}, Success_rate={}".format(result[0],result[1],result[2]))




	# fit on training data notes

	# todo this learner yields ~0.3 success rate
	# logistic_regression_learner = LogisticRegression(solver = "lbfgs",multi_class="multinomial",max_iter = 5000).fit(train,train_labels)

	# todo this learner yields 0.5-0.6 success rate
	# dtree_model = DecisionTreeClassifier(max_depth=7).fit(train, train_labels)



if __name__ == '__main__':
	main()