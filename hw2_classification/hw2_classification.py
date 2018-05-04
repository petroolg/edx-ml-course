from __future__ import division
import numpy as np
import sys

X_train = np.genfromtxt(sys.argv[1], delimiter=",")
y_train = np.genfromtxt(sys.argv[2])[np.newaxis].T
X_test = np.genfromtxt(sys.argv[1], delimiter=",")
y_train = np.genfromtxt(sys.argv[2])[np.newaxis].T

def pluginClassifier(X_train, y_train, X_test):
    class_one_hot = np.array([[y == i for i in range(4)] for y in y_train])
    class_nums = np.sum(class_one_hot.astype(int), axis=0)
    class_priors = class_nums / X_train.shape[0]

    mis, sigmas = [], []

    for i in range(4):
        mi = np.sum(class_one_hot[:, i].astype(int) * X_train, axis=0) / class_nums[i]
        mis.append(mi)

        ss = np.array([np.dot(a[np.newaxis].T, a[np.newaxis]) for a in X_train - mi])
        sigma = np.sum(ss[class_one_hot[:, i].T[0]], axis=0)
        sigmas.append(sigma/class_nums[i])

    probs_list = []
    correct = 0

    for x, y in zip(X_test, y_test):
        probs = []
        for i in range(4):
            probs.append(class_priors[i] / (np.sqrt(2*np.pi*np.linalg.det(sigmas[i]))) * np.exp(-0.5 * (x - mis[i]).dot(np.linalg.inv(sigmas[i])).dot((x - mis[i]).T)))
        probs_list.append(probs/np.sum(probs))
        if np.argmax(probs_list) == y:
            correct += 1

    print("Accuracy of classificator:", correct/len(y_test))

    return probs_list

final_outputs = pluginClassifier(X_train, y_train, X_test)  # assuming final_outputs is returned from function
np.savetxt("probs_test.csv", final_outputs, delimiter=",")  # write output to file
