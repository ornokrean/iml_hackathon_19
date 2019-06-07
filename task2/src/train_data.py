from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, \
	BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from task2.src.data_processor import prepare_data
from sklearn.linear_model import Lasso, Perceptron
import numpy as np
from matplotlib import pyplot as plt

from pandas import DataFrame
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier

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


def test_ada(data,split_ratio=0.3,m_depth = 8,T=120,l_rate=1.1):

	train, test, train_labels, test_labels = split_data(data, split_ratio)

	ab = AdaBoostClassifier(
		DecisionTreeClassifier(max_depth=m_depth),
		n_estimators=T,
		learning_rate=l_rate)

	ab.fit(train, train_labels)
	return get_success_rate(ab,test,test_labels)

def test_KNN(data,split_ratio=0.3,neighbours = 8):
	train, test, train_labels, test_labels = split_data(data, split_ratio)

	knn=KNeighborsClassifier(n_neighbors = neighbours).fit(train, train_labels)

	knn.fit(train, train_labels)
	return get_success_rate(knn,test,test_labels)

def test_random_forest(data,split_ratio=0.3,T=500,random_state=30):
	train, test, train_labels, test_labels = split_data(data, split_ratio)

	rf = RandomForestClassifier(bootstrap=True,max_features=0.2,n_estimators=T,
								criterion='gini',min_samples_leaf=1,min_samples_split=5,
								random_state=random_state)

	rf.fit(train, train_labels)

	return get_success_rate(rf,test,test_labels)

def test_SVM(data,split_ratio=0.3,kernel='linear'):
	train, test, train_labels, test_labels = split_data(data, split_ratio)
	svm = SVC(kernel = kernel, C = 1)
	svm.fit(train, train_labels)
	return get_success_rate(svm,test,test_labels)

def test_logistic_regression(data,split_ratio=0.3,max_iter=4000,solver='lbfgs',
							 multi_class="multinomial"):
	train, test, train_labels, test_labels = split_data(data, split_ratio)
	lr = LogisticRegression(solver=solver, multi_class=multi_class, max_iter=max_iter)
	lr.fit(train,train_labels)
	return get_success_rate(lr,test,test_labels)

def test_ridge_regression(data,split_ratio=0.3,solver='svd',alpha=5):
	train, test, train_labels, test_labels = split_data(data, split_ratio)
	rl = RidgeClassifier(alpha=alpha, normalize=False, solver=solver)
	rl.fit(train,train_labels)
	return get_success_rate(rl,test,test_labels)

def test_ada_boost_with_rf(data,split_ratio=0.3,T=100, random_state = 30):

	train, test, train_labels, test_labels = split_data(data, split_ratio)

	rf = AdaBoostClassifier(
		RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=30),
		n_estimators=T,
		learning_rate=1.1)
	rf.fit(train, train_labels)

	return get_success_rate(rf, test, test_labels)

def test_perceptron(data,split_ratio=0.3):
	train, test, train_labels, test_labels = split_data(data, split_ratio)

	clf = Perceptron(tol=1e-3, random_state=0)

	clf.fit(train,train_labels)

	return get_success_rate(clf,test,test_labels)

def test_ada_perceptron(data,split_ratio=0.3,T=100):
	train, test, train_labels, test_labels = split_data(data, split_ratio)

	ada_clf = AdaBoostClassifier(
		Perceptron(tol=1e-3, random_state=0),
		n_estimators=T,
		learning_rate=1.1)
	ada_clf.fit(train, train_labels)

	return get_success_rate(ada_clf, test, test_labels)

def test_GFC(data,split_ratio=0.3):
	train, test, train_labels, test_labels = split_data(data, split_ratio)

	gfc = GradientBoostingClassifier(n_estimators=20, learning_rate = 1.1, max_features=2,
									 max_depth = 2, random_state = 0)
	gfc.fit(train,train_labels)
	return get_success_rate(gfc,test,test_labels)

def do_bagging(data,classifier,split_ratio=0.3):
	train, test, train_labels, test_labels = split_data(data, split_ratio)

	bc = BaggingClassifier(classifier,n_jobs=4)

	bc.fit(train,train_labels)

	return get_success_rate(bc,test,test_labels)



def main():

	data_amount = 100000
	# get processed data
	# small_data_amount = round(data_amount/10)
	print("data amount:",data_amount)
	# print("small data amount:",small_data_amount)
	# smaller_data = prepare_data(CSV_PATH, small_data_amount)
	data = prepare_data(CSV_PATH, data_amount)
	# print(list(smaller_data))

	# todo this learner yields ??? success rate
	print("bagging classifier (random forest) success rate:",do_bagging(data,
																		RandomForestClassifier(
																			bootstrap=True,
																			max_features=0.2,
																			n_estimators=50,
																			criterion='gini',
																			min_samples_leaf=1,
																			min_samples_split=5,
																			random_state=30)))
	# todo this learner yields ??? success rate
	# print("gfc success rate:",test_GFC(data))
	# todo this learner yields 0.6 success rate
	# print("rf_ada success rate:", test_ada_boost_with_rf(data, 0.3, 100))
	# todo this learner yields 0.3-0.4 success rate
	# print("perceptron success rate:", test_perceptron(data, 0.3))
	# todo this learner yields 0.3-0.4 success rate
	# print("ada perceptron success rate:", test_ada_perceptron(smaller_data, 0.3))
# # todo this learner yields 0.3-0.4 success rate
	# print("ada success rate:",test_ada(smaller_data))
	# # todo this learner yields 0.3-0.4 success rate
	# print("knn success rate:",test_KNN(smaller_data))
	# # todo this learner yields 0.5-0.6 success rate
	# print("Single tree success rate: ",get_tree_success_rate(data,0.3,5))
	# # todo this learner yields 0.5-0.6 success rate
	print("Random forest success rate: ",test_random_forest(data,0.3))
	# # # todo this learner yields ????? success rate
	# print("Logistic regression success rate:",test_logistic_regression(data))
	# # # todo this learner yields ????? success rate
	# print("Ridge regression success rate:", test_ridge_regression(data))



	# # todo this learner yields ????? success rate
	# print("SVM success rate: ", test_SVM(data, 0.3, 'linear'))
	# # todo this learner yields ????? success rate
	# print("SVM success rate: ", test_SVM(data, 0.3, 'polynomial'))


# fit on training data notes

	# todo this learner yields ~0.3 success rate
	# logistic_regression_learner = LogisticRegression(solver = "lbfgs",multi_class="multinomial",max_iter = 5000).fit(train,train_labels)

	# todo this learner yields 0.5-0.6 success rate
	# dtree_model = DecisionTreeClassifier(max_depth=7).fit(train, train_labels)



if __name__ == '__main__':
	main()