# Import Libraries
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# Loading Dataset
iris=datasets.load_iris()
#print(iris.keys())
#dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])
features=iris.data
labels=iris.target

# Training Classifier
clf=KNeighborsClassifier()
clf.fit(features,labels)

pred=clf.predict([[1,1,1,1]])
print(pred)
