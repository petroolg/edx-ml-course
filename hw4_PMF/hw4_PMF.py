from __future__ import division
import numpy as np
import sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# ratings.csv: A comma separated file containing the data. Each row contains a three values that correspond in order to: user_index, object_index, rating
train_data = np.genfromtxt(sys.argv[1], delimiter = ",")

lam = 2
var = 0.1
d = 5


def PMF(train_data):
    data_shape = (int(max(train_data[:, 0])) + 1, int(max(train_data[:, 1])) + 1)

    L, U_matrices, V_matrices = [], [], []

    V = np.random.standard_normal((d, data_shape[1])) / lam
    U = np.zeros((d,data_shape[0])) / lam

    real_M = np.genfromtxt('real_M.csv', delimiter=",")[:data_shape[0], :data_shape[1]]

    for _ in range(50):
        for i, u in enumerate(U.T):
            sigma = np.zeros((d,d))
            Mv = np.zeros((d,1))
            for j in train_data[train_data[:,0]==i][:,1].astype(int):
                sigma += (V[:, j][np.newaxis].T.dot(V[:, j][np.newaxis])).copy()
                Ms = train_data[(train_data[:,0]==i) * (train_data[:,1]==j)][0,2]
                Mv += (V[:,j][np.newaxis].T*Ms).copy()
            ui = np.linalg.inv(lam*var*np.eye(d)+sigma.copy()).dot(Mv.copy())
            U[:,i] = ui.ravel().copy()

        for i, v in enumerate(V.T):
            sigma = np.zeros((d,d))
            Mu = np.zeros((d,1))
            for j in train_data[train_data[:,1]==i][:,0].astype(int):
                sigma += U[:, j][np.newaxis].T.dot(U[:, j][np.newaxis])
                Mu += U[:,j][np.newaxis].T*(train_data[(train_data[:,0]==j) * (train_data[:,1]==i)][0,2])
            vi = np.linalg.inv(lam*var*np.eye(d)+sigma).dot(Mu)
            V[:,i] = vi.ravel().copy()

        U_matrices.append(U.T.copy())
        V_matrices.append(V.T.copy())

        l = 0

        for j in range(train_data.shape[0]):
            res = U[:,int(train_data[j, 0])][np.newaxis].dot(V[:, int(train_data[j, 1])][np.newaxis].T)[0,0]
            l = l - 0.5 / var * (train_data[j, 2] - res) ** 2
        l = l - lam / 2 * np.sum(np.linalg.norm(V, 2, axis=0)) - lam / 2 * np.sum(np.linalg.norm(U, 2, axis=0))

        L.append(l)

    plt.figure()
    plt.imshow(real_M)

    plt.figure()
    plt.imshow(U.T.dot(V))

    plt.figure()
    plt.plot(L)
    plt.show()

    return L, U_matrices, V_matrices


# Implement function here
# def PMF(train_data):
#
#     data_shape = (int(max(train_data[:,0]))+1, int(max(train_data[:,1]))+1)
#
#     L, U_matrices, V_matrices = [], [], []
#
#     v = np.random.standard_normal((d, data_shape[1]))/lam
#     u = np.zeros((data_shape[0], d))/lam
#
#     real_M = np.genfromtxt('real_M.csv', delimiter=",")[:data_shape[0], :data_shape[1]]
#
#     for _ in range(50):
#         for j in range(u.shape[0]):
#             sigma = (train_data[train_data[:,0]==j][:,1]).astype(int)
#             Ssum = np.einsum('nm,nk->mk', v[:,sigma].T,v[:,sigma].T).copy()
#             # Ssum = np.sum([v[:,sigma].T[i][np.newaxis].T.dot(v[:,sigma].T[i][np.newaxis]) for i in range(len(sigma))], axis=0)
#             Vsum = ((v[:,sigma].dot(train_data[sigma,2]))[np.newaxis].T).copy()
#             u[j, :] = np.linalg.inv(lam*var*np.eye(d,d) + Ssum).dot(Vsum.copy()).T[0].copy()
#
#         for j in range(v.shape[1]):
#             sigma = (train_data[train_data[:,1]==j][:,0]).astype(int).copy()
#             Ssum2 = np.einsum('nm,nk->mk', u[sigma,:],u[sigma,:]).copy()
#             Usum = (u[sigma,:].T.dot((train_data[sigma,2]))[np.newaxis].T).copy()
#             v[:, j] = np.linalg.inv(lam*var*np.eye(d,d) + Ssum2).dot(Usum.copy()).T[0].copy()
#
#         U_matrices.append(u.copy())
#         V_matrices.append(v.copy())
#
#         l = 0
#
#         for j in range(train_data.shape[0]):
#             l = l - 0.5/var*(train_data[j, 2] - u[int(train_data[j, 0]), :].dot(v[:,int(train_data[j, 1])]))**2
#         l = l - lam/2*np.sum(np.linalg.norm(u, 2, axis=1)) - lam/2*np.sum(np.linalg.norm(u, 2, axis=0))
#
#         L.append(l)
#
#         # print(i)
#         # print(u.dot(v))
#
#
#     plt.imshow(real_M)
#
#     plt.figure()
#     plt.imshow(u.dot(v))
#
#     plt.figure()
#
#     plt.plot(L)
#     plt.show()
#
#     return L, U_matrices, V_matrices



# Assuming the PMF function returns Loss L, U_matrices and V_matrices (refer to lecture)
L, U_matrices, V_matrices = PMF(train_data)

# np.savetxt("objective.csv", L, delimiter=",")

# np.savetxt("U-10.csv", U_matrices[9], delimiter=",")
# np.savetxt("U-25.csv", U_matrices[24], delimiter=",")
# np.savetxt("U-50.csv", U_matrices[49], delimiter=",")
#
# np.savetxt("V-10.csv", V_matrices[9], delimiter=",")
# np.savetxt("V-25.csv", V_matrices[24], delimiter=",")
# np.savetxt("V-50.csv", V_matrices[49], delimiter=",")