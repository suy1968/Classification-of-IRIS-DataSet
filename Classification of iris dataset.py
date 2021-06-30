from sklearn.datasets import load_iris
iris=load_iris()
print(iris)
print(iris.keys())
print(iris["DESCR"])
print(iris.data)
print(iris.data.T)
import matplotlib .pyplot as plt
print(iris.data.T)
features=iris.data.T
print(features[0])
sepal_length=features[0]
sepal_width=features[1]
petal_length=features[2]
petal_width=features[3]
iris.feature_names
sepal_length_label=iris.feature_names[0]
sepal_width_label=iris.feature_names[1]
petal_length_label=iris.feature_names[2]
petal_width_label=iris.feature_names[3]
print(sepal_length_label)
#print(sepal_width_label)
#print(petal_length_label)
#print(petal_width_label)
plt.scatter(sepal_length,sepal_width,c=iris.target)
plt.xlabel(sepal_length_label)
plt.ylabel(sepal_width_label)
plt.show()
plt.scatter(petal_length,petal_width,c=iris.target)
plt.xlabel(petal_length_label)
plt.ylabel(petal_width_label)
plt.show()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(iris['data'],iris['target'],random_state=0)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
import numpy as np
#X_new=np.array([5,2.9,1,0.2])
X_new=np.array([[5.0,2.9,1.0,0.2]])
print(X_new.shape)
prediction=knn.predict(X_new)
print(prediction)
print(knn.score(X_test,y_test))


