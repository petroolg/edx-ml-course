import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':

    X = np.random.random((1000,2)) - 0.5
    y = np.zeros((1,1000))

    y[0,(X[:, 0] > 0)*(X[:, 1] > 0)] = 1
    y[0,(X[:, 0] > 0)*(X[:, 1] < 0)] = 2
    y[0,(X[:, 0] < 0)*(X[:, 1] > 0)] = 3

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(X[:,0],X[:,1],c=y*10)

    plt.show()

    np.savetxt('X.csv', X, delimiter=',')
    np.savetxt('y.csv', y.astype(int)[np.newaxis].T)