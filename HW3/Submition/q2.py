import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Define the ID3 decision tree class
class DecisionTree:
    def __init__(self, maxdepth=np.inf):
        self.tree = {}
        self.maxdepth = maxdepth

    # Calculate the entropy of a given dataset, the distribution is over the target class.
    def calculate_entropy(self, data):
        labels = data.iloc[:, -1]

        # Your code goes here
        entropy = 0

        number_of_samples = labels.value_counts()
        total_number_of_samples = len(labels)

        for i in number_of_samples:
            P_w = i / total_number_of_samples
            entropy += - P_w * np.log2(P_w)

        return entropy

    # Calculate the information gain of a feature based on its value
    def calculate_information_gain(self, data, feature):
        total_entropy = self.calculate_entropy(data)
        information_gain = total_entropy

        distincts = list(set(data[feature]))  # get the values of the feature

        # Your code goes here
        for value in distincts:
            new_data = self.filter_data(data, feature, value)
            entropy = self.calculate_entropy(new_data)
            information_gain -= (len(new_data) / len(data)) * entropy

        return information_gain

    def filter_data(self, data, feature, value):
        return data[data[feature] == value].drop(feature, axis=1)

    def create_tree(self, data, depth=0):
        # Recursive function to create the decision tree
        labels = data.iloc[:, -1]

        # Base case: if all labels are the same, return the label
        if len(np.unique(labels)) == 1:
            return list(labels)[0]

        features = data.columns.tolist()[:-1]

        # Base case: if there are no features left to split on, return the majority label
        if len(features) == 0:
            unique_labels, label_counts = np.unique(labels, return_counts=True)
            majority_label = unique_labels[label_counts.argmax()]
            return majority_label

        # Base case: if the max depth is reached, return the majority label
        if depth >= self.maxdepth:
            unique_labels, label_counts = np.unique(labels, return_counts=True)
            majority_label = unique_labels[label_counts.argmax()]
            return majority_label

        selected_feature = None
        best_gain = 0

        # Select feature that maximizes gain
        for feature in features:
            gain = self.calculate_information_gain(data, feature)
            if gain > best_gain:
                selected_feature = feature
                best_gain = gain

        # Create the tree node
        tree_node = {}

        distincts = list(set(data[selected_feature]))
        for value in distincts:
            new_data = self.filter_data(data, selected_feature, value)
            if depth < self.maxdepth:
                tree_node[(selected_feature, value)] = self.create_tree(new_data, depth + 1)

        return tree_node

    def fit(self, data):
        self.tree = self.create_tree(data)

    def predict(self, X):
        X = [row[1] for row in X.iterrows()]

        # Predict the labels for new data points
        predictions = []

        for row in X:
            current_node = self.tree
            while isinstance(current_node, dict):
                split_condition = next(iter(current_node))
                feature, value = split_condition
                if (feature, row[feature]) not in current_node.keys():
                    break
                current_node = current_node[feature, row[feature]]
            predictions.append(current_node)

        return predictions

    def _plot(self, tree, indent):
        depth = 1
        for key, value in tree.items():
            if isinstance(value, dict):
                print(" " * indent + str(key) + ":")
                depth = max(depth, 1 + self._plot(value, indent + 2))
            else:
                print(" " * indent + str(key) + ": " + str(value))

        return depth

    def plot(self):
        depth = self._plot(self.tree, 0)
        print(f'depth is {depth}')


data = pd.read_csv('cars.csv')

'''					SECTION A 					'''
tree = DecisionTree()
tree.fit(data)
tree.plot()

'''					SECTION B 					'''
train, test = train_test_split(data, test_size=0.2, random_state=7)

# Get training accuracy
tree = DecisionTree()
tree.fit(train)
pred = tree.predict(train)
acc = (pred == train.iloc[:, -1]).sum() / len(train) * 100
print(f'Training accuracy is {acc}')

'''					SECTION C 					'''
tree = DecisionTree(maxdepth=2)
tree.fit(train)
# Your code goes here
pred = tree.predict(train)
acc = (pred == train.iloc[:, -1]).sum() / len(train) * 100
print(f'Training accuracy is {acc}')

# Your code goes here
pred = tree.predict(test)
acc = (pred == test.iloc[:, -1]).sum() / len(test) * 100
print(f'Test accuracy is {acc}')