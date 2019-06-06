from task1.src.data_processor import *
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn as skl
import seaborn as sns







def main():
	# feces = read_file_into_matrix(small_sambosak)
	feces = read_file_into_matrix(big_sambosak)
	rename_column(feces, {'Unnamed: 0': "FileRow"})
	break_up_date_label(feces, "Date")
	break_up_date_label(feces, "Updated On")
	convert_crime_to_usable_dummy(feces)

	# draw_all_corr_matrices(feces)

	# feces.info()
	# print(feces.iloc[0])

	print(feces.isna().sum())
	print(feces.shape)
	# create_possible_values_file(feces)


if __name__ == '__main__':
	main()
