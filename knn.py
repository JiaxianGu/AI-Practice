class KNearestNeighborsClassifier:
	def __init__(self, K):
		self.K = K

	def fit(self, X_train, Y_train, X_test, Y_test):
		self.X_train = X_train
		self.Y_train = Y_train
		self.X_test = X_test
		self.Y_test = Y_test
		#load all data
	def euclidean_distance(self, point1, point2):
		#calculate euclidean distance between two points
		sum_squared_distance = 0.0
		for i in range(len(point1)-1):
			sum_squared_distance += math.pow((point1[i] - point2[i]), 2)
		return math.sqrt(sum_squared_distance)

	def get_neighbors(self, x):
		# for each test sample, get the most nearest 3 neighbors
		distances = list()
		for sample in self.X_train:
			dist = self.euclidean_distance(sample, x)
			distances.append((sample, dist))
		distances.sort(key=lambda tup: tup[1])
		neighbors = list()
		for i in range(self.K):
			# print(distances[i][0])
			neighbors.append(distances[i][0])
		return neighbors

	def predict_classification(self, test_sample):
		# of the most nearest 3 neighbors, decide the class that most samples belong to
		neighbors = self.get_neighbors(test_sample)
		output_values = [row[-1] for row in neighbors]
		prediction = max(set(output_values), key=output_values.count)
		return prediction

	def predict(self, X_test):
		#predict all element of test data
		X_test_predict = list()
		for row in X_test:
			output = self.predict_classification(row)
			
			X_test_predict.append(output)
		return X_test_predict


if __name__ == '__main__':
	import math
	import numpy as np
	from sklearn.datasets import load_iris
	from sklearn.model_selection import train_test_split
	iris_dataset = load_iris()
	X_train, X_test, Y_train, Y_test = train_test_split(iris_dataset.data,iris_dataset.target,random_state=0)
	Y_train_knn = Y_train.reshape(Y_train.shape[0],1)
	Y_test_knn = Y_test.reshape(Y_test.shape[0],1)
	X_train_knn = np.append(X_train, Y_train_knn, axis = 1)
	X_test_knn = np.append(X_test, Y_test_knn, axis = 1)
	#combine target and data together, so that the prediction algorithm works
	knn = KNearestNeighborsClassifier(3)
	knn.fit(X_train_knn, Y_train_knn, X_test_knn, Y_test_knn)

	X_test_predict = knn.predict(X_test_knn)
	accuracy = np.sum(Y_test == X_test_predict) / Y_test.shape[0]
	print(accuracy)
	assert(accuracy > 0.7)
	print('acc:', accuracy)