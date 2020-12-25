import numpy as np
from from_scratch.collaborative_filter import collaborative_filter
from from_scratch.knn import KNN
from from_scratch.distances import euclidean_distances, manhattan_distances, cosine_distances
from from_scratch.import_data import train_test_split
from csv import reader

with open("IRIS.csv", 'r') as file:
    csv_reader = reader(file)
    feature_names = list(next(csv_reader))
    data = np.array(list(csv_reader))

feature_names = feature_names[:-1]
features = data[:,:-1].astype("float").T
targets = data[:,-1]
targets = targets.reshape((1, targets.shape[0]))
train_features, train_targets, test_features, test_targets = train_test_split(features, targets)

model = KNN(n_neighbors=3)
model.fit(train_features, train_targets)
predictions = model.predict(test_features)

print("Accuracy = " + str(100 * np.sum(predictions == test_targets)/predictions.shape[1]) + "%")