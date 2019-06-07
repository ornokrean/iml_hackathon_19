import pickle
import numpy as np
CLASS_HEADER = 'Primary Type'
from task2.src.data_processor import prepare_data, align_columns
import pandas as pd

def get_learner():
	# loaded_learner = pickle.load(open("learner_data.sav",'rb'))
	loaded_learner = pd.read_hdf('learner_data.h5','d1')
	return loaded_learner

def get_headers():
	# loaded_headers = pickle.load(open("headers_data.sav",'rb'))
	loaded_headers = pd.read_hdf('headers_data.h5','d1')

	return loaded_headers

def align_columns(known_headers, test_data):
	for header in known_headers:
		if header not in test_data.columns.values:
			test_data[header] = np.zeros(test_data.shape[0])
	test_data.pop(CLASS_HEADER)

def classify(data):
	data = prepare_data(data, False)
	known_headers = get_headers()
	align_columns(known_headers,data)
	learner = get_learner()
	out = learner.predict(data)
	return out