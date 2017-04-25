import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier

#load and print iris
iris= load_iris()
print(iris.data)


print("Resubstitution Validation (pre-pruning depth<3 with boosting&bagging)\n")
dtc= BaggingClassifier(AdaBoostClassifier( tree.DecisionTreeClassifier(max_depth=3)))
r= dtc.fit(iris.data, iris.target)
print("Score: ", r.score(iris.data, iris.target))
pred= dtc.predict(iris.data)
print(pred)
cnf= confusion_matrix(iris.target, pred)
print(cnf)


print("Resubstitution Validation (pre-pruning depth<3)\n")
dtc= tree.DecisionTreeClassifier(max_depth=3)
r= dtc.fit(iris.data, iris.target)
print("Score: ", r.score(iris.data, iris.target))
pred= dtc.predict(iris.data)
print(pred)
cnf= confusion_matrix(iris.target, pred)
print(cnf)
with open("iris.dot","w") as pic:
          pic= tree.export_graphviz(dtc,out_file=pic)


print("\nK-fold cross validation (pre-pruning depth<3)")
print("\n[K-Fold K=2]")
kf2= KFold(n_splits=2, shuffle=True)

normal= []

for train, test in kf2.split(iris.data, iris.target):
    
    dtc= tree.DecisionTreeClassifier(max_depth=3)
    r= dtc.fit(iris.data[train], iris.target[train])
    print("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\")
    score = r.score(iris.data[test], iris.target[test])
    print("Classification Accuracy = ", score)
    normal.append(score)
    pred= dtc.predict(iris.data[test])
    print(pred)
    cnf= confusion_matrix(iris.target[test], pred)
    print(cnf)

print("|||||||||||||||||||||||||||")
print("Overall Classification accuracy= ", np.asarray(normal).mean())

print("\n[K-Fold K=5]")
kf5= KFold(n_splits=5, shuffle=True)

normal = []

for train, test in kf5.split(iris.data, iris.target):
    
    dtc= tree.DecisionTreeClassifier(max_depth=3)
    r= dtc.fit(iris.data[train], iris.target[train])
    print("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\")
    score= r.score(iris.data[test], iris.target[test])
    print("Classification Accuracy = ", score)
    normal.append(score)
    pred= dtc.predict(iris.data[test])
    print(pred)
    cnf= confusion_matrix(iris.target[test], pred)
    print(cnf)

print("|||||||||||||||||||||||||||")
print("Overall Classification accuracy= ", np.asarray(normal).mean())

print("\n[K-Fold K=10]")
kf10= KFold(n_splits=10, shuffle=True)

normal = []

for train, test in kf10.split(iris.data, iris.target):
    dtc= tree.DecisionTreeClassifier(max_depth=3)
    r= dtc.fit(iris.data[train], iris.target[train])
    #print("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\")
    #print(pred)
    score= r.score(iris.data[test], iris.target[test])
    print("Classification Accuracy = ", score)
    normal.append(score)
    pred= dtc.predict(iris.data[test])
    cnf= confusion_matrix(iris.target[test], pred)
    #print(cnf)
    with open("iris10.dot","w") as pic:
          pic= tree.export_graphviz(dtc,out_file=pic)

print("|||||||||||||||||||||||||||")
print("Overall Classification accuracy= ", np.asarray(normal).mean())

print("\n[K-Fold K=10] with boosting&bagging")
kf20= KFold(n_splits=10, shuffle=True)
normal = []

for train, test in kf20.split(iris.data, iris.target):
    dtc= BaggingClassifier(AdaBoostClassifier( tree.DecisionTreeClassifier(max_depth=3)))
    r=dtc.fit(iris.data[train], iris.target[train])
    #print("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\")
    #print(pred)
    score= r.score(iris.data[test], iris.target[test])
    print("Classification Accuracy = ", score)
    normal.append(score)
    pred= dtc.predict(iris.data[test])
    cnf= confusion_matrix(iris.target[test], pred)
    #print(cnf)
    

print("|||||||||||||||||||||||||||")
print("Overall Classification accuracy= ", np.asarray(normal).mean())

