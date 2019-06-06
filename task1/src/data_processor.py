import os,sys

import pandas as pd
import numpy as np
import matplotlib as plt
import sklearn as skl


CSV_PATH = "tweets_data"



def read_file_into_matrix(path):
	pd_df = pd.read_csv(path)
	return pd_df



def main():
	path = os.path.join(CSV_PATH,'cristiano_tweets.csv')
	pd_df = read_file_into_matrix(path)

	pd_df.info()







if __name__=='__main__':
	main()