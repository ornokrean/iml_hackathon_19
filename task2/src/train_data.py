import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from task2.src.data_processor import prepare_data
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier


CSV_PATH = "Crimes_since_2005.csv"
# CSV_PATH = "partial_data"
CLASS_HEADER = 'Primary Type'

def split_data(data):
	labels = data.pop(CLASS_HEADER)
	return train_test_split(data, labels, test_size=0.3)



def get_error_rate(learner,test, test_labels):
	prediction_fails = learner.predict(test)==test_labels

	return prediction_fails

def logistic_regression(train, labels):
	logistic_regression_learner = LogisticRegression(solver = "lbfgs",multi_class="multinomial",
													 max_iter = 2000).fit(train,labels)
	return logistic_regression_learner

def ridge_regression(train, labels, alpha, solver):
	ridge_learner = RidgeClassifier(alpha=alpha, normalize=False, solver=solver).fit(train, labels)

	return ridge_learner

def KNeighbors(train, labels, n_neighbors):
	neigh = KNeighborsRegressor(n_neighbors=n_neighbors)
	neigh.fit(train, labels)

	return neigh

def lasso_regression(train, labels, alpha):
	lasso = Lasso(alpha=alpha, normalize=False)
	lasso.fit(train, labels)

	return lasso

def random_forest(train, labels, n_samples):
	random_forest = RandomForestClassifier(n_estimators=n_samples, max_depth=2, random_state=0)
	random_forest.fit(train, labels)

	return random_forest

# def plot_error(samples):
# 	plt.figure(1)
# 	for i in range(samples.shape[0]):



def main():
	# get processed data
	data = prepare_data(CSV_PATH,30000)

	# split the data into training, validation and test
	train, test, train_labels, test_labels = split_data(data)

	# fit on training data
	logistic_learner = logistic_regression(train, train_labels)
	#
	alpha = 1
	# solver = 'auto'
	# ridge_learner = ridge_regression(train, train_labels, alpha, solver)

	# lasso_learner = lasso_regression(train, train_labels, alpha)

	random_forest_learner = random_forest(train, train_labels, 1000)


	# validate using validation data

	# test on test data
	test_regression = 1 - logistic_learner.score(test,test_labels)
	# test_ridge = 1 - ridge_learner.score(test, test_labels)
	# test_lasso = 1 - lasso_learner.score(test, test_labels)
	test_random_forest = 1 - random_forest_learner.score(test, test_labels)


	print("test regression error:", test_regression ,'\n', "test ridge error:" ,
		   '\n', "test random forest: ", test_random_forest)

	# ltest = list(test)
	# test_regression = get_error_rate(logistic_learner,test,test_labels)
	# print("Success rate: ",end="")
	# print(ltest.count(True)/float(len(ltest)))

if __name__=='__main__':
	main()