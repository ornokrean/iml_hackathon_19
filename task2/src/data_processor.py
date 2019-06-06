import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
from matplotlib.colors import ListedColormap

CSV_PATH = "Crimes_since_2005.csv"
# CSV_PATH = "partial.csv"
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


def rename_column(data, dict):
	data.rename(columns=dict, inplace=True)

def break_up_date_label(data, label):
	#	05/27/2019 11:50:00 PM
	#	0123456789012345678901
	date_col = pop_column(data, label)
	# date_col = data[label]
	data[label + '_year']= date_col.apply(lambda row: int(row[6:10]))
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


def plot_corr(df,size=10):
	corr = df.corr()
	fig, ax = plt.subplots(figsize=(size, size))
	ax.matshow(corr)
	plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical')
	plt.yticks(range(len(corr.columns)), corr.columns)
	plt.show()


def print_heatmap(df):
	corr = df.corr()
	sns.heatmap(corr,
			xticklabels=corr.columns.values,
			yticklabels=corr.columns.values)
	plt.show()


def plot_better_corr(df):
	labels = df.where(np.triu(np.ones(df.shape)).astype(np.bool))
	labels = labels.round(2)
	labels = labels.replace(np.nan,' ', regex=True)

	mask = np.triu(np.ones(df.shape)).astype(np.bool)
	ax = sns.heatmap(df, mask=mask, cmap='RdYlGn_r', fmt='', square=True, linewidths=1.5)
	mask = np.ones((8, 8))-mask
	ax = sns.heatmap(df, mask=mask, cmap=ListedColormap(['white']),annot=labels,cbar=False, fmt='', linewidths=1.5)
	ax.set_xticks([])
	ax.set_yticks([])
	plt.show()


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
		if label in ['District', 'Location Description']:
			continue
		mean = train[str(label)].mean()
		train[str(label)].fillna(mean, inplace=True)


def main() -> None:
	# Import data
	pd_df = read_file_into_matrix(CSV_PATH)
	rename_column(pd_df, {'Unnamed: 0': "FileRow"})
	# print(pd_df.info())

	# Print number of samples of each type
	print_num_of_unique_samples_of_each_label(pd_df)

	# Split data to train and test
	labels = pd_df['Primary Type']
	train, test, train_labels, test_labels = initial_data_split(pd_df, labels)

	# Drop columns
	train = train.drop(columns=['Primary Type'])

	# Split dates
	break_up_date_label(train, "Date")
	break_up_date_label(train, "Updated On")

	# Change True/False to ints
	train = train.applymap(lambda x: 1 if x is True else x)
	train = train.applymap(lambda x: 0 if x is False else x)

	# Change Block column to numeric values
	encode_block_column(train)

	# Fill all nans with mean
	fill_nans_with_mean(train)

	# Get categorical columns
	train = split_to_categories(train, ['District', 'Location Description'])


# # Describe
# print_describe(pd_df)


# # Print all crimes types
# print_all_crimes_types(train_labels)
#

# # Plot correlation
	# plot_corr(pd_df)
	# print_heatmap(pd_df)
	# plot_better_corr(pd_df)




if __name__ == '__main__':
	main()
