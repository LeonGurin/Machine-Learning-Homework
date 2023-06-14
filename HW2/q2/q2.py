import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from numpy.linalg import eig
from sklearn.metrics import accuracy_score
import os

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        cov_matrix = np.cov(X.T)
        eigenvalues, eigenvectors = eig(cov_matrix)

        self.components = eigenvectors.T[:self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]

    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components.T)

    def inverse_transform(self, X):
        return np.dot(X, self.components) + self.mean

def plot_cdf(data):
    sorted_data = np.sort(data)[::-1]

    data_cumsum = np.cumsum(sorted_data)
    data_normalized = data_cumsum / data_cumsum[-1]

    plt.plot(np.arange(1, len(sorted_data)+1), data_normalized)
    plt.xlabel('Principal Component')
    plt.ylabel('Cumulative Proportion of Variance')
    plt.title('Cumulative Distribution Function of Eigenvalues')
    plt.show()

def plot_images(original, reduced, recovered):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original.reshape(28, 28), cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(reduced.reshape(9, 9), cmap='gray')
    axes[1].set_title('Image after PCA')
    axes[1].axis('off')

    axes[2].imshow(recovered.reshape(28, 28), cmap='gray')
    axes[2].set_title('Recovered Image')
    axes[2].axis('off')

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

def knn(train_X, train_y, test_X, test_y, k, batch_size=100):
    num_test_samples = test_X.shape[0]
    num_batches = int(np.ceil(num_test_samples / batch_size))

    test_preds = np.zeros(num_test_samples)

    for i in range(num_batches):
        print(f"Batch {i+1}/{num_batches}")
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_test_samples)

        test_X_batch = test_X[start_idx:end_idx]

        dist_matrix = distance_matrix(test_X_batch, train_X)
        neighbors_indices = np.argpartition(dist_matrix, k, axis=1)[:, :k]

        neighbors_labels = train_y[neighbors_indices].astype(int)
        test_preds[start_idx:end_idx] = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=neighbors_labels)

    return test_preds

train_X, train_y, test_X, test_y = load_data("fashion-mnist_train.csv", "fashion-mnist_test.csv")

pca = PCA(n_components=81)
pca.fit(train_X)
train_X_reduced = pca.transform(train_X)
test_X_reduced = pca.transform(test_X)

plot_images(train_X[0], train_X_reduced[0], pca.inverse_transform(train_X_reduced[0]))

plt.show()

plot_cdf(pca.explained_variance)

optimal_dim = 100
pca_optimal = PCA(n_components=optimal_dim)
pca_optimal.fit(train_X)
train_X_optimal = pca_optimal.transform(train_X)
test_X_optimal = pca_optimal.transform(test_X)

# k_values = range(1, 15)
# k_values = [10]
# accuracies = [knn(train_X_optimal, train_y, test_X_optimal, test_y, k) for k in k_values]
# optimal_k = k_values[np.argmax(accuracies)]

test_preds = knn(train_X_optimal, train_y, test_X_optimal, test_y, 10)
accuracy = accuracy_score(test_y, test_preds)
print(f"Test accuracy is: {accuracy * 100}%")