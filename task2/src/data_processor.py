import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
from matplotlib.colors import ListedColormap


# CSV_PATH = "Crimes_since_2005.csv"
CSV_PATH = "partial_data"
CLASS_HEADER = 'Primary Type'
SEPARATOR = "************************************************************************************"
small_sambosak = "t.csv"

big_sambosak = "Crimes_since_2005.csv"

crime_dict = {"THEFT":0,
			  "BATTERY":1,
			  "NARCOTICS":2,
			  "BURGLARY":3,
			  "WEAPONS VIOLATION":4,
			  "DECEPTIVE PRACTICE":5,
			  "CRIMINAL TRESPASS":6,
			  "PROSTITUTION":7}



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

def convert_crime_to_usable_dummy(data):
	data[CLASS_HEADER]=data[CLASS_HEADER].apply(lambda val: crime_dict[val])


def rename_column(data, dict):
	data.rename(columns=dict, inplace=True)


def get_data_val_count_list(data, val=None):
	if val == None:
		num_of_batariot = data.value_counts()
	else:
		num_of_batariot = data.value_counts()[val]
	return num_of_batariot


def pop_column(data, column_header):
	return data.pop(column_header)


def get_corr_matrix(data):
	return data.corr()


def draw_heatmap(corr,title=""):
	plt.axes().set_title('Heatmap of the correlation between original cleaned data ('+title+')')
	sns.heatmap(corr, linewidths=1.5, linecolor='black', cmap='RdBu', center=0, vmax=1, vmin=-1)
	plt.show()


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


def create_correleration_matrices(data):
	out_list = []
	for key in crime_dict.keys():
		out_list.append((key,data.loc[data[CLASS_HEADER] == crime_dict[key]]))
	return out_list

def draw_all_corr_matrices(data):
	for mat in create_correleration_matrices(data):
		correlation_matrix = get_corr_matrix(mat[1])
		draw_heatmap(correlation_matrix,mat[0])
	correlation_matrix = get_corr_matrix(data)
	draw_heatmap(correlation_matrix,"all")


def create_possible_values_file(data):
	out_f = open("possible_values.txt",'w')
	# headers = list(data)
	headers = ['Primary Type','Location Description','Arrest','Domestic','District','Ward',
			   'Community Area','Beat','Year', 'Wards']
	for header in headers:
		vals = data[header].unique()
		out_f.write(header+":\n"+str(list(vals))+"\n\n"+SEPARATOR+"\n\n")
		print(header+":\n\t",end="")
		print(vals)
	out_f.close()

def fill_nans_with_mean(train):
	for label in train.columns.values:
		if label in ['District', 'Location Description']:
			continue
		mean = train[label].mean()
		train[label].fillna(mean, inplace=True)

def take_care_of_na(data):
	fill_nans_with_mean(data)
	return data.dropna(how='any')


def prepare_data(data_path,amount):
	data = read_file_into_matrix(data_path).sample(amount)
	# Change True/False to ints
	data = data.applymap(lambda x: 1 if x == True else 0 if x==False else x)

	# Change Block column to numeric values
	encode_block_column(data)
	# Get categorical columns
	data = split_to_categories(data, ['District','Location Description'])
	rename_column(data, {'Unnamed: 0': "FileRow"})
	break_up_date_label(data, "Date")
	break_up_date_label(data, "Updated On")
	convert_crime_to_usable_dummy(data)

	data = take_care_of_na(data)
	return data


def main() -> None:
	pd_df = prepare_data(CSV_PATH)
	print(pd_df.head(5))
	# Split data to train and test
	labels = pd_df['Primary Type']
	train, test, train_labels, test_labels = initial_data_split(pd_df, labels)
	# # Remove Type column
	train = train.drop(columns=['Primary Type'])




if __name__=='__main__':
	main()
