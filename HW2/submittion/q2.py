import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score
import os
import timeit as time
from joblib import Parallel, delayed

def plot_cdf(data):
	sorted_data = np.sort(data)[::-1]

	data_cumsum = np.cumsum(sorted_data)
	data_normalized = data_cumsum / data_cumsum[-1]

	# Plot the CDF of eigenvalues
	plt.plot(np.arange(1, len(sorted_data)+1), data_normalized)
	plt.xlabel('Principal Component')
	plt.ylabel('Cumulative Proportion of Variance')
	plt.title('Cumulative Distribution Function of Eigenvalues')
	plt.show()

def load_data(train_file, test_file):
    # Load data from files.
    # Check that paths exist.
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print("dataset files not found")
        raise FileNotFoundError
    train_data = np.loadtxt(train_file, delimiter=',', skiprows=1)
    test_data = np.loadtxt(test_file, delimiter=',', skiprows=1)

    train_X, train_y = train_data[:, 1:], train_data[:, 0].astype(np.int8)
    test_X, test_y = test_data[:, 1:], test_data[:, 0].astype(np.int8)

    return train_X, train_y, test_X, test_y

def performPCA(data, k=9, plot=False):
	mu = np.mean(data, axis=0)

	# subtract the mean from the data
	Z = data - mu

	S = np.matmul(Z.T, Z)
	eigenvalues, eigenvectors = np.linalg.eig(S)

	# sort the eigenvalues and eigenvectors in descending order
	idx = np.argsort(eigenvalues)[::-1]
	eigenvalues = eigenvalues[idx]
	eigenvectors = eigenvectors[:, idx]

	# plot the CDF of eigenvalues
	# if plot:
	# 	plot_cdf(eigenvalues)

	# select the top k^2 eigenvectors
	E = eigenvectors[:, :(k ** 2)].T
	y = np.matmul(E, Z.T)

	# pick a random image and reconstruct it
	idx = np.random.randint(0, data.shape[0])
	original = data[idx]
	PCA_image = np.matmul(E, (original - mu))
	reconstructed = np.matmul(E.T, PCA_image) + mu

	original = original.reshape(28, 28)
	PCA_image = PCA_image.reshape(k, k)
	reconstructed = reconstructed.reshape(28, 28)
	
	# plot the original and reconstructed images
	if plot:
		fig, ax = plt.subplots(1, 3)
		ax[0].imshow(original, cmap='gray')
		ax[0].set_title('Original Image')
		ax[1].imshow(PCA_image, cmap='gray')
		ax[1].set_title('PCA Image')
		ax[2].imshow(reconstructed, cmap='gray')
		ax[2].set_title('Reconstructed Image')
		plt.show()

	return y.T, E

def performKNN(train_X, train_y, test_X, test_y, k, batch_size=100):
    # Separate the data into batches to avoid memory issues
    num_test_samples = test_X.shape[0]
    num_batches = int(np.ceil(num_test_samples / batch_size))

    test_preds = np.zeros(num_test_samples)

    for i in range(num_batches):
        # print(f"Batch {i+1}/{num_batches}")
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_test_samples)

        test_X_batch = test_X[start_idx:end_idx]

		# compute the distance matrix between the test and train data
        dist_matrix = cdist(test_X_batch, train_X)
	
		# find the k nearest neighbors
        neighbors_indices = np.argpartition(dist_matrix, k, axis=1)[:, :k]

		# find the most common label among the k nearest neighbors
        neighbors_labels = train_y[neighbors_indices]
	
		# assign the most common label to the test sample
        test_preds[start_idx:end_idx] = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=neighbors_labels)

    return test_preds

def knn_batch(train_X, train_y, test_X_batch, k):
    dist_matrix = cdist(test_X_batch, train_X)
    neighbors_indices = np.argpartition(dist_matrix, k, axis=1)[:, :k]
    neighbors_labels = train_y[neighbors_indices]
    return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=neighbors_labels)

def knn_parallel(train_X, train_y, test_X, test_y, k, batch_size=100, n_jobs=-1):
	num_test_samples = test_X.shape[0]
	num_batches = int(np.ceil(num_test_samples / batch_size))

	test_preds = Parallel(n_jobs=n_jobs)(delayed(knn_batch)(train_X, train_y, test_X[i * batch_size: (i + 1) * batch_size], k) for i in range(num_batches))
	print(f"k = {k}, accuracy = {accuracy_score(test_y, np.concatenate(test_preds))}")
	return np.concatenate(test_preds)

if __name__ == '__main__':
	# start timer	
	start = time.default_timer()

	# read the data
	train_data, train_labels, test_data, test_labels = load_data('fashion-mnist_train.csv', 'fashion-mnist_test.csv')

	# use PCA to reduce dimensionality from 28 by 28 to 9 by 9
	PCA_train_data = performPCA(train_data, k=9, plot=False)
	
	# test the model on the test data
	optimal_dim_PCA = 10
	PCA_train_data, E = performPCA(train_data, k=optimal_dim_PCA, plot=False)
	PCA_test_data = np.matmul(E, (test_data - np.mean(test_data, axis=0)).T).T

	# commented out
	# k_values = range(1, 15)
	# test_preds = [knn_parallel(PCA_train_data, train_labels, PCA_test_data, test_labels, k) for k in k_values]
	# accuracy = [accuracy_score(test_labels, test_pred) for test_pred in test_preds]
	# optimal_k_KNN = k_values[np.argmax(accuracy)]
	# print(f"Optimal k for KNN is: {optimal_k_KNN}")

	optimal_k_KNN = 9
	# test_preds = knn_parallel(PCA_train_data, train_labels, PCA_test_data, test_labels, k=optimal_k_KNN, batch_size=100)
	test_preds = performKNN(PCA_train_data, train_labels, PCA_test_data, test_labels, k=optimal_k_KNN, batch_size=100)
	accuracy = accuracy_score(test_labels, test_preds)
	print(f"Test accuracy is: {accuracy * 100}%")

	# stop timer
	stop = time.default_timer()
	# print('Time: ', stop - start)

