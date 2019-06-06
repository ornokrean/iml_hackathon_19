from sklearn.model_selection import train_test_split

from task2.src.data_processor import prepare_data

from sklearn.linear_model import LogisticRegression

CSV_PATH = "Crimes_since_2005.csv"
# CSV_PATH = "partial_data"
CLASS_HEADER = 'Primary Type'




def split_data(data):
	labels=data.pop(CLASS_HEADER)
	return train_test_split(data, labels, test_size=0.3)





def main():
	# get processed data
	data = prepare_data(CSV_PATH,15000)

	# split the data into training, validation and test
	train, test, train_labels, test_labels = split_data(data)
	print("original data:",data.shape)
	print("train data:",train.shape,"train_labels:",train_labels.shape)
	print("test data:",test.shape,"test_labels:",test_labels.shape)


	# fit on training data
	logistic_regression_learner = LogisticRegression(solver = "lbfgs",multi_class="multinomial",
													 max_iter = 5000).\
		fit(train,train_labels)
	# validate using validation data
	print(logistic_regression_learner.score(test,test_labels))
	# test on test data

if __name__=='__main__':
	main()