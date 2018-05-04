from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# #############################################################################
# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1], [-1, 1], [0, 0]]
X, labels_true = make_blobs(n_samples=500, centers=centers, cluster_std=0.2,
                            random_state=0)

X = StandardScaler().fit_transform(X)

plt.scatter(X[:,0], X[:,1])
plt.show()

np.savetxt('X.csv', X, delimiter=',')