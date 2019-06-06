import os,sys

import pandas as pd
import numpy as np
import matplotlib as plt
import sklearn as skl


CSV_PATH = "tweets_data"



def read_file_into_matrix(path):
	pd_df = pd.read_csv(path)
	return pd_df

def get_tweet_matrices(folder_path):
	matrices = []
	for file in os.listdir(folder_path):
		matrices.append(read_file_into_matrix(os.path.join(folder_path,file)))
	return matrices

def main():
	matrices = get_tweet_matrices(CSV_PATH)
	for pd_df in matrices:

		pd_df.info()







if __name__=='__main__':
	main()