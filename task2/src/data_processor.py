import os,sys

import pandas as pd
import numpy as np
import matplotlib as plt
import sklearn as skl


CSV_PATH = "/cs/usr/davidnir1/IML_HACK/iml-hackathon-2019/crimes_data/Crimes_since_2005.csv"



def read_file_into_matrix(path):
	pd_df = pd.read_csv(path)



def main():
	pd_df = read_file_into_matrix(CSV_PATH)
	print(pd_df)







if __name__=='__main__':
	main()