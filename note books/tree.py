import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("data\\Merged file1.csv")

specified_country = data[data["Year"] == int(input("Enter your target year: "))]

X = specified_country["Oil consumption - TWh"]
y = specified_country["Cluster"]
X = np.array([int(i) for i in X]).reshape(-1, 1)
y = np.array([int(i) for i in y]).reshape(-1, 1)

clf = DecisionTreeClassifier(max_depth=3).fit(X, y)

plt.figure(figsize=(10, 6))
plot_tree(clf, filled=True, feature_names=y, class_names=[str(i) for i in clf.classes_])
print(clf.predict(X))
plt.show()
