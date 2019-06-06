import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
from matplotlib.colors import ListedColormap

# CSV_PATH = "Crimes_since_2005.csv"
CSV_PATH = "partial.csv"

pd.set_option("display.max_rows", 20000, "display.max_columns", 90, "display.width", 100)


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


def main() -> None:
	# Import data
	pd_df = read_file_into_matrix(CSV_PATH)
	# print(pd_df.info())

	# Print number of samples of each type
	print_num_of_unique_samples_of_each_label(pd_df)

	# Split data to train and test
	labels = pd_df['Primary Type']
	train, test, train_labels, test_labels = initial_data_split(pd_df, labels)

	# Remove Type column
	train = train.drop(columns=['Primary Type'])

	# Change True/False to ints
	train = train.applymap(lambda x: 1 if x == True else x)
	train = train.applymap(lambda x: 0 if x == False else x)

	# Change Block column to numeric values
	encode_block_column(train)

	# Get categorical columns
	train = split_to_categories(train, ['District','Location Description'])
	print(train.head(50))


#
# # Describe
# print_describe(pd_df)


# # Print all crimes types
# print_all_crimes_types(train_labels)
#



if __name__=='__main__':
	main()
