import numpy as np

usersN = 10
moviesN = 6

users = range(usersN)

M = []
for u in users:
    movies = np.random.choice(range(moviesN), np.random.randint(1, 6), replace=False)
    ratings = np.random.choice(range(5),len(movies))
    M.append(np.vstack((np.repeat(u,len(movies)), movies, ratings)).T)

M = np.vstack(M)
real_M = np.zeros((usersN, moviesN))
real_M[M[:,0],M[:,1]] = M[:,2]

np.savetxt('M.csv', M, fmt=['%d, ']*3)
np.savetxt('real_M.csv', real_M, fmt=['%d,']*moviesN)
