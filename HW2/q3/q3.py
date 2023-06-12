import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def plot(data, labels, w):
    fig, ax = plt.subplots()

    c0 = data[labels == 0]
    c1 = data[labels == 1]

    ax.scatter(c0[:,0], c0[:,1], c='red')
    ax.scatter(c1[:,0], c1[:,1], c='blue')
    
    a, b, c = w
    m = -a / b
    b = -c / b

    x = np.arange(np.min(data[:,0]), np.max(data[:,0]), 0.1)
    y = m * x + b
    plt.plot(x, y)

    plt.show()

def read_data(filename):
    # read the dataset from a csv file
    data = np.genfromtxt(filename, delimiter=',')
    # split into features and labels
    X = data[:,:-1]
    y = data[:,-1]
    return X, y

def sigmoid(z):
    return np.clip(1 / (1 + np.exp(-z)), 1e-6, 1 - 1e-6)

def loss(y, y_pred):
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

def Logistic_Regression_via_GD(P, y, lr):
    n, d = P.shape
    w = np.zeros(d)
    b = 0
    
    num_iterations = 200000
    for i in range(num_iterations):
        # Calculate the predictions
        z = np.dot(P, w) + b
        y_pred = sigmoid(z)

        # Calculate the gradients for weights and bias
        dw = np.dot(P.T, (y_pred - y)) / n
        db = np.sum(y_pred - y) / n

        # Update the weights and bias
        w -= lr * dw
        b -= lr * db
            
        # if i % 10000 == 0:
        #     print(f'Iteration {i}, loss: {loss(y, y_pred)}')

    return w, b

def Predict(w, b, p):
    # Predict the labels for the data points in p
    z = np.dot(p, w) + b
    y_pred = sigmoid(z)
    return np.round(y_pred)

def main():
    # read the data
    X, y = read_data('exams.csv')

    avg_accuracy = 0
    best_accuracy = 0
    best_w, best_b = None, None
    for i in range(10):
        # split into training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        # train the model
        w, b = Logistic_Regression_via_GD(X_train, y_train, 0.0015)
        # Predict labels for the test set
        y_pred = Predict(w,b,X_test)
        # calculate the accuracy
        accuracy = accuracy_score(y_test, y_pred)
        avg_accuracy += accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_w, best_b = w, b
    avg_accuracy /= 10
    print(f"Avg test accuracy: {avg_accuracy * 100}%")
    # plot the data and the decision boundary
    plot(X, y, [best_w[0], best_w[1], best_b])

if __name__ == '__main__':
    main()
