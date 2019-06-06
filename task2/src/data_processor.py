import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

EXCLUDE_FROM_NAN = ['District', 'Location Description', 'Primary Type']

CSV_PATH = "Crimes_since_2005.csv"
#CSV_PATH = "partial.csv"
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

def convert_crime_to_usable_dummy(data):
	data[CLASS_HEADER]=data[CLASS_HEADER].apply(lambda val: crime_dict[val])


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


def initial_data_split(data, labels):
	return train_test_split(data, labels, test_size=0.3)


def print_all_crimes_types(train_labels):
	print('All crimes types:')
	labels = train_labels.unique()
	print(labels)
	print()


def print_num_of_unique_samples_of_each_label(pd_df):
	print('Number of samples of each type:')
	for label in pd_df.columns.values:
		num_of_diff_values = pd_df[label].unique()
		print("Number of unique values of label \"" + str(label) + "\" is: " + str(len(num_of_diff_values)))
	print()


def print_describe(pd_df):
	print("Describe:")
	describe = pd_df.describe()
	print(describe)
	print()


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





def remove_lowest_correlation1(data):

	y = data.pop(CLASS_HEADER)
	# apply SelectKBest class to extract top 10 best features
	bestfeatures = SelectKBest(score_func=chi2, k=5)
	fit = bestfeatures.fit(data, y)
	dfscores = pd.DataFrame(fit.scores_)
	dfcolumns = pd.DataFrame(data.columns)
	# concat two dataframes for better visualization
	# featureScores = pd.concat([dfcolumns, dfscores], axis=1)
	# featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
	# print(featureScores.nlargest(10, 'Score'))  # print 10 best features

	data = data[dfcolumns]
	data[CLASS_HEADER] = y
	return data




def remove_lowest_correlation(data):
	labels = data.pop(CLASS_HEADER)
	# Create correlation matrix
	corr_matrix = data.corr().abs()

	# Select upper triangle of correlation matrix
	upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
	# Find index of feature columns with correlation greater than 0.95
	to_drop = [column for column in upper.columns if any(upper[column] > 0.2)]
	for i in to_drop:
		data.pop(i)
	# data.drop(data.columns[to_drop],axis=1,inplace=True)
	data[CLASS_HEADER]=labels
	return data

def get_david_hypothesis(data):
	y = data.pop(CLASS_HEADER)
	print(y.value_counts())
	data[CLASS_HEADER] = y

def plot_long_lat(data):
	# vec = pd.concat([data['Latitude'],data['Longitude']])
	# vec1 = data['Latitude']
	# vec2 = data['Longitude']
	vec = data[['Latitude','Longitude']]
	print(vec.values)
	print(vec.shape)
	vec.plot(kind="scatter",x='Latitude',y='Longitude')

	plt.ylim(-88,-87.5)
	plt.xlim(41.6, 42.1)
	plt.show()
	exit(-1)

def prepare_data(path, num_of_samples):
	# Import data
	data = read_file_into_matrix(path).sample(num_of_samples)

	# Drop columns
	data = data.drop(columns=['Unnamed: 0', 'ID', 'Updated On', 'Community Areas'])

	# Split dates
	break_up_date_label(data, "Date")
	# break_up_date_label(data, "Updated On")

	# Change True/False to ints
	data = data.applymap(lambda x: 1 if x is True else 0 if x is False else x)

	# Change Block column to numeric values
	encode_block_column(data)

	# Fill all nans with mean
	fill_nans_with_mean(data)

	# cos_cos = lambda x, y: np.cos(x) * np.cos(y)
	# cos_sin = lambda x, y: np.cos(x) * np.sin(y)
	# sin = lambda x: np.sin(x)
	# funca = np.vectorize(cos_cos)
	# funcb = np.vectorize(cos_sin)
	# funcc = np.vectorize(sin)
	# data["x"] = funca(data["Latitude"], data["Longitude"])
	# data["y"] = funcb(data["Latitude"], data["Longitude"])
	# data["z"] = funcc(data["Latitude"])
	# data = data.drop(columns=['Latitude', 'Longitude', 'X Coordinate', 'Y Coordinate','Wards'])
	# Get categorical columns
	data = split_to_categories(data, ['District', 'Location Description', 'Year'])

	# data = remove_lowest_correlation(data)

	# get_david_hypothesis(data)

	plot_long_lat(data)

	return data


if __name__ == '__main__':
	print("Go sell kebab")