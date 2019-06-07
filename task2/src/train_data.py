from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, \
	BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from task2.src.classifier import classify
from task2.src.data_processor import prepare_data, align_columns
from sklearn.linear_model import Perceptron
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
import pandas as pd


CSV_PATH = "Crimes_since_2005.csv"
CLASS_HEADER = 'Primary Type'


def read_file_into_matrix(path):
	pd_df = pd.read_csv(path)
	return pd_df

def split_data(data, ratio):
	labels = data.pop(CLASS_HEADER)
	split_result = train_test_split(data, labels, test_size=ratio)
	data[CLASS_HEADER] = labels
	return split_result


def get_success_rate(learner, test, test_labels):
	prediction_success_indices = learner.predict(test) == test_labels
	ltest = list(prediction_success_indices)
	return round(ltest.count(True) / float(len(ltest)), 3)


def test_ada(data, split_ratio=0.3, m_depth=8, T=120, l_rate=1.1):
	train, test, train_labels, test_labels = split_data(data, split_ratio)

	ab = AdaBoostClassifier(
		DecisionTreeClassifier(max_depth=m_depth),
		n_estimators=T,
		learning_rate=l_rate)

	ab.fit(train, train_labels)
	return get_success_rate(ab, test, test_labels)


def test_KNN(data, split_ratio=0.3, neighbours=8):
	train, test, train_labels, test_labels = split_data(data, split_ratio)

	knn = KNeighborsClassifier(n_neighbors=neighbours).fit(train, train_labels)

	knn.fit(train, train_labels)
	return get_success_rate(knn, test, test_labels)


def test_random_forest(data, split_ratio=0.3, T=500, random_state=30):
	train, test, train_labels, test_labels = split_data(data, split_ratio)

	rf = RandomForestClassifier(bootstrap=True, max_features=0.2, n_estimators=T,
								criterion='gini', min_samples_leaf=1, min_samples_split=5,
								random_state=random_state)

	rf.fit(train, train_labels)

	return get_success_rate(rf, test, test_labels)


def test_SVM(data, split_ratio=0.3, kernel='linear'):
	train, test, train_labels, test_labels = split_data(data, split_ratio)
	svm = SVC(kernel=kernel, C=1)
	svm.fit(train, train_labels)
	return get_success_rate(svm, test, test_labels)


def test_logistic_regression(data, split_ratio=0.3, max_iter=4000, solver='lbfgs',
							 multi_class="multinomial"):
	train, test, train_labels, test_labels = split_data(data, split_ratio)
	lr = LogisticRegression(solver=solver, multi_class=multi_class, max_iter=max_iter)
	lr.fit(train, train_labels)
	return get_success_rate(lr, test, test_labels)


def test_ridge_regression(data, split_ratio=0.3, solver='svd', alpha=5):
	train, test, train_labels, test_labels = split_data(data, split_ratio)
	rl = RidgeClassifier(alpha=alpha, normalize=False, solver=solver)
	rl.fit(train, train_labels)
	return get_success_rate(rl, test, test_labels)


def test_ada_boost_with_rf(data, split_ratio=0.3, T=100, random_state=30):
	train, test, train_labels, test_labels = split_data(data, split_ratio)

	rf = AdaBoostClassifier(
		RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=30),
		n_estimators=T,
		learning_rate=1.1)
	rf.fit(train, train_labels)

	return get_success_rate(rf, test, test_labels)


def test_perceptron(data, split_ratio=0.3):
	train, test, train_labels, test_labels = split_data(data, split_ratio)

	clf = Perceptron(tol=1e-3, random_state=0)

	clf.fit(train, train_labels)

	return get_success_rate(clf, test, test_labels)


def test_ada_perceptron(data, split_ratio=0.3, T=100):
	train, test, train_labels, test_labels = split_data(data, split_ratio)

	ada_clf = AdaBoostClassifier(
		Perceptron(tol=1e-3, random_state=0),
		n_estimators=T,
		learning_rate=1.1)
	ada_clf.fit(train, train_labels)

	return get_success_rate(ada_clf, test, test_labels)


def test_GFC(data, split_ratio=0.3):
	train, test, train_labels, test_labels = split_data(data, split_ratio)

	gfc = GradientBoostingClassifier(n_estimators=20, learning_rate=1.1, max_features=2,
									 max_depth=2, random_state=0)
	gfc.fit(train, train_labels)
	return get_success_rate(gfc, test, test_labels)


def do_bagging(data, classifier, split_ratio=0.3):
	train, test, train_labels, test_labels = split_data(data, split_ratio)

	bc = BaggingClassifier(classifier, n_jobs=4)

	bc.fit(train, train_labels)

	return get_success_rate(bc, test, test_labels)


def save_learner_for_later(learner):
	hdf = pd.HDFstore("learner_data.h5")
	hdf.put('d1',learner,format="table",data_columns=True)
	hdf.close()
	# pickle.dump(learner, open("learner_data.sav", 'wb'))

def save_headers_for_later(data):
	headers = data.columns.values
	# pickle.dump(headers,open("headers_data.sav",'wb'))
	hdf = pd.HDFstore("headers_data.h5")
	hdf.put('d1',headers,format="table",data_columns=True)
	hdf.close()


def fit_learner_and_save_as_file(data):
	train, test, train_labels, test_labels = split_data(data, 0.3)

	bc = BaggingClassifier(RandomForestClassifier(bootstrap=True, max_features=0.2,
												  n_estimators=50, criterion='gini',
												  min_samples_leaf=1, min_samples_split=5,
												  random_state=30))

	bc.fit(train, train_labels)
	save_headers_for_later(data)
	save_learner_for_later(bc)
	print("our learner got success rate of:", get_success_rate(bc, test, test_labels))
	return bc


def main():
	data = prepare_data(CSV_PATH,True)
	# test_data = prepare_data(CSV_PATH, True)
	test_data = read_file_into_matrix("partial_data.csv")
	# align_columns(data, test_data)

	learner = fit_learner_and_save_as_file(data)
	# print("before\n",learner.predict(test_data))
	restored_result = classify(test_data)
	print("after\n",restored_result)




if __name__ == '__main__':
	main()
