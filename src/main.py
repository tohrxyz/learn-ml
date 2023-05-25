from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics

iris = datasets.load_iris()

print("Features: ", iris.feature_names)

print("Labels: ", iris.target_names)

print(iris.data[:5])

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)

clf = tree.DecisionTreeClassifier()

clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
