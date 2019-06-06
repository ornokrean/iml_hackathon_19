from task2.src.data_processor import prepare_data

# CSV_PATH = "Crimes_since_2005.csv"
CSV_PATH = "partial_data"



def main():
	data = prepare_data(CSV_PATH)
	print(data)

if __name__=='__main__':
	main()