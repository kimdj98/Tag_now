import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime

#----- import data starts -----#
import os
if not os.path.exists('user2movie.json') or \
   not os.path.exists('movie2user.json') or \
   not os.path.exists('usermovie2rating.json') or \
   not os.path.exists('usermovie2rating_test.json'):
   import preprocess2dict

with open('user2movie.json', 'rb') as f:
    user2movie = pickle.load(f)

with open('movie2user.json', 'rb') as f:
    movie2user = pickle.load(f)

with open('usermovie2rating.json', 'rb') as f:
    usermovie2rating = pickle.load(f)

with open('usermovie2rating_test.json', 'rb') as f:
    usermovie2rating_test = pickle.load(f)

#----- import data ends -----#

N = np.max(list(user2movie.keys())) + 1
# the test set may contain movies the train set doesn't have data on
m1 = np.max(list(movie2user.keys()))
m2 = np.max([m for (u, m), r in usermovie2rating_test.items()])
M = max(m1, m2) + 1

# initialize variables
K = 10 # latent dimensionality
W = np.random.randn(N,K) # user factorization term
b = np.zeros(N) # user bias
U = np.random.randn(M,K) # movie factorization term
c = np.zeros(M) # movie bias
mu = np.mean(list(usermovie2rating.values())) # total variance

# to show how losses decrease
def get_loss(d):
    # d: (user_id, movie_id) -> rating:
    N = float(len(d))
    sse = 0
    for k, r in d.items():
        i, j = k
        p = W[i].dot(U[j]) + b[i] + c[j] + mu
        sse += np.square(p-r)
    return sse / N

# train the parameters
epochs = 25
reg = 0.01 # regularization penalty
# to show how losses decrease
train_losses = []
test_losses = []
for epoch in range(epochs):
    print(f'epoch: {epoch}')
    epoch_start = datetime.now()
    # perform updates

    # update W and b
    t0 = datetime.now()
    for i in range(N):
        # for W
        matrix = np.eye(K) * reg
        vector = np.zeros(K)

        # for b
        bi = 0
        for j in user2movie[i]: # implementation of xai_i
            r = usermovie2rating[(i,j)]
            matrix += np.outer(U[j], U[j])
            vector += (r - b[i] - c[j] - mu) * U[j]
            bi += (r - W[i].dot(U[j]) - c[j] - mu)
        
        # set the updates
        W[i] = np.linalg.solve(matrix, vector)
        b[i] = bi / (len(user2movie[i]) + reg)

        if i % (N // 10) == 0:
            print(f'i: {i} /N: {N}')
    print(f"updated W and b: {datetime.now() - t0}")
    # update U and c
    for j in range(M):
        # for U
        matrix = np.eye(K) * reg
        vector = np.zeros(K)

        # for c
        cj = 0
        try:
            for i in movie2user[j]:
                r = usermovie2rating[(i,j)]
                matrix += np.outer(W[i], W[i])
                vector += (r - b[i] - c[j] - mu) * W[i]

                cj += (r - np.dot(W[i], U[j]) - b[i] - mu)

            if j % (M // 10) == 0:
                print(f'j: {j} /M: {M}')
            # set the updates
            U[j] = np.linalg.solve(matrix, vector)
            c[j] = cj / (len(movie2user[j]) + reg)

        except KeyError:
            # possible not to have any ratings for a movie
            pass
    print(f'updated U and c: {datetime.now() - t0}')
    print(f'epoch duration: {datetime.now() - epoch_start}')        

    # store train loss
    t0 = datetime.now()
    train_losses.append(get_loss(usermovie2rating))

    # store test loss
    test_losses.append(get_loss(usermovie2rating_test))
    print('calculate cost:', datetime.now() - t0)
    print('train loss:', train_losses[-1])
    print('test_loss:', test_losses[-1])

print('train losses:', train_losses)
print('test_losses:', test_losses)

plt.plot(train_losses, label = 'train loss')
plt.plot(test_losses, label = 'test loss')
plt.legend()
plt.show()