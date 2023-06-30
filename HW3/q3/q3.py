import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt

# Copy here your full decision tree from q1

class RandomForest:
	def __init__(self, maxdepth=np.inf, n_estimators=3, method='simple'):
		self.forest = []
		self.maxdepth = maxdepth
		self.n_estimators = n_estimators
		self.method = method

	def select_features(self, data):
		np.random.seed(40+len(self.forest))
		
		if self.method == 'sqrt':
			m = int(np.sqrt(len(data.columns)-1))
		elif self.method == 'log':
			m = int(np.log2(len(data.columns)-1))
		else:
			m = np.random.randint(0, len(data.columns))

		incidies = np.random.choice(np.arange(0, len(data.columns)-1), size=m, replace=False)
		features = list(data.columns[incidies])
		return data[features + ['class']]

	def fit(self, data):
		self.forest = []
		# Your code goes here

	def _predict(self, X):
		# Predict the labels for new data points
		predictions = []

		preds = [tree.predict(X) for tree in self.forest]
		preds = list(zip(*preds))
		predictions = [Counter(est).most_common(1)[0][0] for est in preds]

		return predictions
		
	def score(self, X, sample_weight=None):
		pred = self._predict(X)
		return (pred == X.iloc[:,-1]).sum() / len(X)


def KFold(data, model, cv=5):
	chunks = np.array_split(data, cv)
	correct_sum = 0
	# Your code goes here
	return correct_sum / cv


data = pd.read_csv('cars.csv')

train, test = train_test_split(data, test_size=0.2, random_state=13)

'''					SECTION B 					'''
# Run single decision tree:
# tree = DecisionTree(maxdepth=3)
# tree.fit(train)

# pred = tree.predict(train)
# acc = (pred == train.iloc[:,-1]).sum() / len(train)
# print(f'Decision Tree Training accuracy is {acc}')

# pred = tree.predict(test)
# acc = (pred == test.iloc[:,-1]).sum() / len(test)
# print(f'Decision Tree Test accuracy is {acc}')
# print()

# Run random forest
# forest = RandomForest(maxdepth=3, n_estimators=5, method='simple')
# forest.fit(train)

# acc = forest.score(train)
# print(f'Random Forest Training accuracy is {acc}')

# acc = forest.score(test)
# print(f'Random Forest Test accuracy is {acc}')

'''					SECTION C 					'''

# correct = []

# for i in range(1,8):
	# forest = RandomForest(maxdepth=3, n_estimators=i, method='simple')
	# correct.append(KFold(data=train, model=forest, cv=5))

# plt.plot(range(1,8), np.array(correct))
# plt.xlabel('trees num')
# plt.ylabel('avg accuracy')
# plt.show()

