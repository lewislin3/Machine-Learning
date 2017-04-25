import numpy as np
import pandas as pd
import csv
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.neighbors import KDTree
from sklearn.neighbors.classification import KNeighborsClassifier
from timeit import default_timer as timer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold




data= np.genfromtxt('winequality-white.csv', delimiter=";", skip_header=1,usecols=(0,1,2,3,4,5,6,7,8,9,10))
target= np.genfromtxt('winequality-white.csv', delimiter=";", skip_header=1,usecols=(11))


print(data)
print(target)


print("\nKD_tree euclidean\n")
clf = KNeighborsClassifier(n_neighbors=2,algorithm='kd_tree',metric='euclidean')
kf = KFold(n_splits=5, shuffle=True)

for train, test in kf.split(data, target):
    train_time = timer()
    r=clf.fit(data[train], target[train])
    train_time = timer()-train_time
    test_time = timer()
    pred = clf.predict(data[test])
    test_time = timer()-test_time
    cnf_matrix = confusion_matrix(target[test], pred)
    print(cnf_matrix)
    score= r.score(data[test], target[test])
    print("Classification Accuracy = ", score)
    print("Train elapsed time =", train_time)
    print("Test elapsed time =", test_time)

print("\nKD_tree manhattan\n")
clf = KNeighborsClassifier(n_neighbors=2,algorithm='kd_tree',metric='manhattan')
kf = KFold(n_splits=5, shuffle=True)

for train, test in kf.split(data, target):
    train_time = timer()
    r=clf.fit(data[train], target[train])
    train_time = timer()-train_time
    test_time = timer()
    pred = clf.predict(data[test])
    test_time = timer()-test_time
    cnf_matrix = confusion_matrix(target[test], pred)
    print(cnf_matrix)
    score= r.score(data[test], target[test])
    print("Classification Accuracy = ", score)
    print("Train elapsed time =", train_time)
    print("Test elapsed time =", test_time)








print("\nbrute euclidean\n")
clf = KNeighborsClassifier(n_neighbors=2,algorithm='brute',metric='euclidean')
kf = KFold(n_splits=5, shuffle=True)

for train, test in kf.split(data, target):
    train_time = timer()
    r=clf.fit(data[train], target[train])
    train_time = timer()-train_time
    test_time = timer()
    pred = clf.predict(data[test])
    test_time = timer()-test_time
    cnf_matrix = confusion_matrix(target[test], pred)
    print(cnf_matrix)
    score= r.score(data[test], target[test])
    print("Classification Accuracy = ", score)
    print("Train elapsed time =", train_time)
    print("Test elapsed time =", test_time)


print("\nbrute manhattan\n")
clf = KNeighborsClassifier(n_neighbors=2,algorithm='brute',metric='manhattan')
kf = KFold(n_splits=5, shuffle=True)

for train, test in kf.split(data, target):
    train_time = timer()
    r=clf.fit(data[train], target[train])
    train_time = timer()-train_time
    test_time = timer()
    pred = clf.predict(data[test])
    test_time = timer()-test_time
    cnf_matrix = confusion_matrix(target[test], pred)
    print(cnf_matrix)
    score= r.score(data[test], target[test])
    print("Classification Accuracy = ", score)
    print("Train elapsed time =", train_time)
    print("Test elapsed time =", test_time)


print("\nbrute cosine\n")
clf = KNeighborsClassifier(n_neighbors=2,algorithm='brute',metric='cosine')
kf = KFold(n_splits=5, shuffle=True)

for train, test in kf.split(data, target):
    train_time = timer()
    r=clf.fit(data[train], target[train])
    train_time = timer()-train_time
    test_time = timer()
    pred = clf.predict(data[test])
    test_time = timer()-test_time
    cnf_matrix = confusion_matrix(target[test], pred)
    print(cnf_matrix)
    score= r.score(data[test], target[test])
    print("Classification Accuracy = ", score)
    print("Train elapsed time =", train_time)
    print("Test elapsed time =", test_time)
