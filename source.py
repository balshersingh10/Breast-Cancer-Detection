import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import classification_report, accuracy_score
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import pandas as pd

#url="E:\ML\Projects\Breast Cancer Detection\breast-cancer-wisconsin.data"
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
names=['id', 'clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',
       'marginal_adhesion', 'single_epithelial_size', 'bare_nuclei',
       'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']
df = pd.read_csv(url, names=names)
df.replace('?',-99999,inplace=True)
print(df.axes)
#print(df)
#print(df.shape)
df.drop(['id'],1,inplace=True)
print(df)
print(df.shape)
print(df.loc[6])
print(df.describe())
df.hist(figsize = (10, 10))
plt.show()
scatter_matrix(df,figsize=(18,18))
plt.show()


X=np.array(df.drop(['class'],1))
y=np.array(df['class'])
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

models=[]
models.append(('KNN',KNeighborsClassifier(n_neighbors=6)))
models.append(('SVM',SVC(kernel='linear')))
#models.append(('SVM',SVC(kernel='rbf')))
#models.append(('SVM',SVC(kernel='poly')))
#models.append(('SVM',SVC(kernel='sigmoid')))

results = []
names = []

for name,model in models:
    kfold=model_selection.KFold(n_splits=10, random_state = 0, shuffle=True)
    cv_results=model_selection.cross_val_score(model,X_train,y_train,cv=kfold,scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print("{}:accuracy->{}(std->{})".format(name, cv_results.mean(), cv_results.std()))

for name, model in models:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(name)
    print(accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))

clf = SVC(kernel='linear')

clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measures = np.array([[4,1,3,5,2,1,4,1,10]])
#print(example_measures)
example_measures = example_measures.reshape(len(example_measures), -1)
#print(example_measures)
prediction = clf.predict(example_measures)
print(prediction)
