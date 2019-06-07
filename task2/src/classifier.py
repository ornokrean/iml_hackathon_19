import pickle

def get_learner():
	loaded_learner = pickle.load(open("learner_data.sav",'rb'))
	return loaded_learner

def classify(data):
	learner = get_learner()
	out = learner.predict(data)
	return out