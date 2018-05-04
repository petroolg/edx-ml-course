import numpy as np
import pandas as pd
import scipy as sp
import sys

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse

N_clusters = 5
N_iterations = 15


def normPDF(x, mean, var):

    denom = (2*np.pi*np.linalg.det(var))**.5
    num = np.exp(-0.5*(x-mean).dot(np.linalg.inv(var)).dot((x-mean).T))
    return num/denom

def plotEllipse(mean, var, ax):

    eigs = np.linalg.eigvals(np.nan_to_num(var))

    ell = Ellipse(xy=(mean[0], mean[1]),
                  width=eigs[0] * 2, height=eigs[1] * 2,
                  angle=np.rad2deg(np.arccos(var[0, 0])),
                  facecolor='none',
                  edgecolor='r')
    ax.add_artist(ell)


def KMeans(X):

    # find bounding box of data
    max_v = np.max(X, axis=0)
    min_v = np.min(X, axis=0)
    mean = np.mean(X, axis=0)
    scale = max_v - min_v

    # initialization of centroids
    x_dim = X.shape[1]
    centroids = (np.random.random((5, x_dim)) - 0.5) * scale + mean

    for j in range(N_iterations):

        plt.scatter(X[:, 0], X[:, 1])
        plt.scatter(centroids[:, 0], centroids[:, 1])
        plt.waitforbuttonpress()
        plt.cla()

        clusters = [[] for _ in range(N_clusters)]

        # samples assignment
        for x in X:
            dists = (centroids - x)**2
            dists = np.sum(dists,axis=1)
            cluster = np.argmin(dists)
            clusters[cluster].append(x.copy())

        # centroids update
        for i in range(N_clusters):
            centroids[i] = np.mean(clusters[i], axis=0)

        # filename = "centroids-" + str(j + 1) + ".csv"  # "i" would be each iteration
        # np.savetxt(filename, centroids, delimiter=",")

    plt.close()

def EMGMM(X):

    # find bounding box of data
    max_v = np.max(X, axis=0)
    min_v = np.min(X, axis=0)
    mean = np.mean(X, axis=0)
    scale = max_v - min_v

    x_dim = X.shape[1]
    mix_coeffs = np.ones((N_clusters))/N_clusters
    sigmas = [0.5*np.eye(x_dim, x_dim) for _ in range(N_clusters)]
    means = (np.random.random((5, x_dim)) - 0.5) * scale + mean

    for j in range(N_iterations):

        resps = [[] for _ in range(N_clusters)]

        f, ax = plt.subplots()
        ax.scatter(X[:, 0], X[:, 1])

        for i in range(N_clusters):
            plotEllipse(means[i], sigmas[i], ax)

        mngr = plt.get_current_fig_manager()
        mngr.window.wm_geometry("+500+0")

        f.waitforbuttonpress()
        plt.close()
        # plt.show()

        # Evaluate the responsibilities
        for i in range(N_clusters):
            for x in X:
                resps[i].append(mix_coeffs[i]*normPDF(x,means[i],sigmas[i]))
        resps = (np.array(resps)/np.sum(resps, axis=0)).T

        for i in range(N_clusters):


            means[i] = resps[:,i].T.dot(X)/np.sum(resps[:,i])

            sigma = np.eye(x_dim, x_dim)

            for r, x in zip(resps[:,i], X):
                sigma = sigma + r*(x-means[i])[np.newaxis].T.dot((x-means[i])[np.newaxis])
            sigmas[i] = sigma/np.sum(resps[:,i])

            mix_coeffs[i] = np.sum(resps[:,i])/X.shape[0]

            # filename = "Sigma-" + str(i + 1) + "-" + str(
            #     j + 1) + ".csv"  # this must be done 5 times (or the number of clusters) for each iteration
            # np.savetxt(filename, sigmas[i], delimiter=",")

        # filename = "pi-" + str(j+1) + ".csv"
        # np.savetxt(filename, mix_coeffs, delimiter=",")
        # filename = "mu-" + str(j+1) + ".csv"
        # np.savetxt(filename, means, delimiter=",")  #this must be done at every iteration


X = np.genfromtxt(sys.argv[1], delimiter=",")
KMeans(X)
EMGMM(X)