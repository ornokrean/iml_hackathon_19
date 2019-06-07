import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.cluster import KMeans
import numpy as np

EXCLUDE_FROM_NAN = ['District', 'Location Description', 'Primary Type']

CSV_PATH = "Crimes_since_2005.csv"
CLASS_HEADER = 'Primary Type'
pd.set_option("display.max_rows", 20000, "display.max_columns", 90, "display.width", 100)

crime_dict = {"THEFT": 0,
			  "BATTERY": 1,
			  "NARCOTICS": 2,
			  "BURGLARY": 3,
			  "WEAPONS VIOLATION": 4,
			  "DECEPTIVE PRACTICE": 5,
			  "CRIMINAL TRESPASS": 6,
			  "PROSTITUTION": 7}


def break_up_date_label(data, label):
	# 05/27/2019 11:50:00 PM
	# 0123456789012345678901
	date_col = pop_column(data, label)
	# date_col = data[label]
	# data[label + '_year']= date_col.apply(lambda row: int(row[6:10]))
	data[label + '_day'] = date_col.apply(lambda row: int(row[3:5]))
	data[label + '_month'] = date_col.apply(lambda row: int(row[0:2]))
	data[label + '_hour'] = date_col.apply(lambda row:
										   (int(row[11:13]) if row[20:22] == 'AM' else int(
											   row[11:13]) + 12) + int(row[14:16]) / 60.0)

def pop_column(data, column_header):
	return data.pop(column_header)

def read_file_into_matrix(path):
	pd_df = pd.read_csv(path)
	return pd_df

def split_to_categories(train, features):
	for feature in features:
		dummy = pd.get_dummies(train[feature])
		train = pd.concat([train, dummy], axis=1)
	train = train.drop(columns=features)
	return train

def encode_block_column(train):
	enc = OrdinalEncoder()
	blocks = train['Block'].values
	train['Block'] = enc.fit_transform(blocks.reshape(-1, 1))


def fill_nans_with_mean(train):
	for label in train.columns.values:
		if label in EXCLUDE_FROM_NAN:
			continue
		median = train[label].median()
		train[label].fillna(median, inplace=True)

colors = {"THEFT": "red",
		  "BATTERY": "green",
		  "NARCOTICS": "blue",
		  "BURGLARY": "pink",
		  "WEAPONS VIOLATION": "purple",
		  "DECEPTIVE PRACTICE": "orange",
		  "CRIMINAL TRESPASS": "black",
		  "PROSTITUTION": "brown"}


def align_columns(data, test_data):
	for header in data.columns.values:
		if header not in test_data.columns.values:
			test_data[header] = np.zeros(test_data.shape[0])
	for header in test_data.columns.values:
		if header not in data.columns.values:
			data[header] = np.zeros(data.shape[0])
	test_data.pop(CLASS_HEADER)



def prepare_data(path,read_file):
	# Import data

	data = read_file_into_matrix(path).sample(300000) if read_file else path

	if not read_file:
		print(path.columns.values)

	# Drop columns
	for header in ['Unnamed: 0', 'ID', 'Updated On', 'Community Areas']:
		if header in data.columns.values:
			pop_column(data,header)
	# data = data.drop(columns=['Unnamed: 0', 'ID', 'Updated On', 'Community Areas'])

	# Split dates
	break_up_date_label(data, "Date")

	# Change True/False to ints
	data = data.applymap(lambda x: 1 if x is True else 0 if x is False else x)
	crimes = pop_column(data,CLASS_HEADER)
	data[CLASS_HEADER] = crimes.map(lambda x: crime_dict[x])

	# Change Block column to numeric values
	encode_block_column(data)

	# Fill all nans with mean
	fill_nans_with_mean(data)

	data = split_to_categories(data, ['District', 'Location Description', 'Year'])
	# sample_num = data.shape[0]
	sample_num = len(data[CLASS_HEADER].unique())
	cluster_num = sample_num if sample_num < 8 else 8
	kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(data[['Latitude', 'Longitude']])

	data['cluster'] = kmeans.labels_

	data = data.drop(columns=['Latitude', 'Longitude', 'X Coordinate', 'Y Coordinate', 'Wards'])

	return data

