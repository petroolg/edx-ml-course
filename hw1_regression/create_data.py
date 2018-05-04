import numpy as np
from matplotlib import pyplot as plt


X = np.random.random((100,1))*5
X = np.hstack((X, np.ones_like(X)))
Y = X.dot([[0.3], [2]])
Y += (np.random.random(Y.shape)-0.5)*0.3

plt.scatter(X[:,0],Y)
plt.show()

np.savetxt('X_train.csv', X, delimiter=',')
np.savetxt('y_train.csv', Y)
np.savetxt('X_test.csv', X, delimiter=',')