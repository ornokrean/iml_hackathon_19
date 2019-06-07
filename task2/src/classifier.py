import numpy as np
CLASS_HEADER = 'Primary Type'
from .data_processor import prepare_data
from sklearn.externals import joblib

def get_learner():
	return joblib.load('learner_data.pkl')


def get_headers():
	return joblib.load('headers_data.pkl')

def align_columns(known_headers, test_data):
	for header in known_headers:
		if header not in test_data.columns.values:
			test_data[header] = np.zeros(test_data.shape[0])

	test_data.pop(CLASS_HEADER)

def classify(data):
	known_headers = get_headers()

	data = prepare_data(data, False)
	align_columns(known_headers,data)

	for header in data.columns.values:
		if header not in known_headers:
			data.drop(columns=[header])
	learner = get_learner()
	out = learner.predict(data)
	return out