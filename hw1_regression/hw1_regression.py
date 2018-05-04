import numpy as np
import sys

lambda_input = int(sys.argv[1])
sigma2_input = float(sys.argv[2])
X_train = np.genfromtxt(sys.argv[3], delimiter = ",")
y_train = np.genfromtxt(sys.argv[4])
X_test = np.genfromtxt(sys.argv[5], delimiter = ",")

## Solution for Part 1
def part1():
    w = np.linalg.inv(X_train.T.dot(X_train)+lambda_input*np.eye(X_train.shape[1])).dot(X_train.T).dot(y_train)
    return w

## Solution for Part 2
def part2(X_test, X_train):

    Xtest = X_test.copy()
    Xtrain = X_train.copy()
    indices = []
    sigma2 = Xtrain[:,-1].T.dot(Xtrain[:,-1])/sigma2_input + lambda_input * np.eye(Xtrain.shape[1]-1)
    for j in range(10):

        ind = 0
        val_max = 0
        for i, x in enumerate(Xtest):
            val = sigma2_input + x[:-1][np.newaxis].dot(np.linalg.inv(sigma2)).dot(x[:-1][np.newaxis].T)
            if val > val_max and (i+1) not in indices:
                val_max = val
                ind = i+1
        indices.append(int(ind))
        sigma2 = sigma2 + Xtest[ind-1,:-1][np.newaxis].T.dot(Xtest[ind-1,:-1][np.newaxis])
    return indices



wRR = part1()  # Assuming wRR is returned from the function
np.savetxt("wRR_" + str(lambda_input) + ".csv", wRR, delimiter="\n") # write output to file

active = part2(X_test, X_train)  # Assuming active is returned from the function
file = open("active_" + str(lambda_input) + "_" + str(int(sigma2_input)) + ".csv", 'w+')
file.write(','.join([str(a) for a in active]))
file.close()

