import os,sys

import pandas as pd
import numpy as np
import matplotlib as plt
import sklearn as skl
import seaborn


CSV_PATH = "Crimes_since_2005.csv"



def read_file_into_matrix(path):
	pd_df = pd.read_csv(path)

	return pd_df



def main():
	pd_df = read_file_into_matrix(CSV_PATH)

	print(pd_df.info())







if __name__=='__main__':
	main()