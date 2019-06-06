from task1.src.data_processor import *
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn as skl
import seaborn as sns

small_sambosak = "t.csv"

big_sambosak = "Crimes_since_2005.csv"


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


def draw_heatmap(corr):
	plt.axes().set_title('Heatmap of the correlation between original cleaned data')
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


#	hour = (orig_hour if am else orig_hour+12) + orig_min/60


def main():
	feces = read_file_into_matrix(small_sambosak)
	# feces = read_file_into_matrix(big_sambosak)
	rename_column(feces, {'Unnamed: 0': "FileRow"})

	Y = feces['Primary Type']
	# Y = pop_column(feces,'Primary Type')
	# feces.convert_objects(convert_numeric=True)
	crime_appear_count = get_data_val_count_list(Y)
	break_up_date_label(feces, "Date")
	break_up_date_label(feces, "Updated On")
	print(list(feces))
	correlation_matrix = get_corr_matrix(feces)
	draw_heatmap(correlation_matrix)
	print(correlation_matrix)
	print(feces.iloc[0])
	print(feces.iloc[1])


# print(list(feces.ix[[2]]))

# todo for i in headers
#	print(i[1])ג
# feces.info()


if __name__ == '__main__':
	main()