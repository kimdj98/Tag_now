## using vectorized matrix factorization

import pickle
from queue import Empty
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

print('converting...')
user2movierating = {}
for i, movies in user2movie.items():
    r = np.array([usermovie2rating[(i,j)] for j in movies])
    user2movierating[i] = (movies, r)
movie2userrating = {}
for j, users in movie2user.items():
    r = np.array([usermovie2rating[i, j] for i in users])
    movie2userrating[j] = (users, r)

# create a movie2user for test set, since we need it for loss
movie2userrating_test = {}
for (i, j), r in usermovie2rating_test.items():
    if j not in movie2userrating_test:
        movie2userrating_test[j] = [[i], [r]]
    else:
        movie2userrating_test[j][0].append(i)
        movie2userrating_test[j][1].append(r)
for j, (user, r) in movie2userrating_test.items():
    movie2userrating_test[j][1] = np.array(r)
print('conversion done')

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

# to show how losses decrease in later plot
def get_loss(m2u):
    # d: (movie_id) -> (user_ids, ratings):
    N = 0.
    sse = 0
    for j, (user_ids, ratings) in m2u.items():
        
        predictions = W[user_ids].dot(U[j]) + b[user_ids] + c[j] + mu # prediction 값
        sse += np.sum((predictions -ratings)**2) # sum of squares error
        N += len(ratings)
    return sse / N

# train the parameters
epochs = 25
reg = 0.01 # regularization penalty
# to show how losses decrease
train_losses = []
test_losses = []
for epoch in range(epochs):
    print(f'-----------------------------------------------------------------------------------')
    print(f'epoch: {epoch}')
    epoch_start = datetime.now()
    # perform updates

    # update W and b
    t0 = datetime.now()
    for i in range(N):
        movie_ids, ratings = user2movierating[i]
        matrix = np.dot(U[movie_ids].T, U[movie_ids]) + np.eye(K) * reg
        vector = np.dot(ratings - b[i] - c[movie_ids] - mu, U[movie_ids])
        bi = np.sum(ratings - U[movie_ids].dot(W[i]) - c[movie_ids] - mu)

        W[i] = np.linalg.solve(matrix, vector)
        b[i] = bi / (len(movie_ids) + reg)
    
        if i % (N//10) == 0:
            print(f'i: {i}, N: {N}')
    print(f'updated W and b: {datetime.now() - t0}')

    # update U and c
    t0 = datetime.now()
    for j in range(M):
        try:
            user_ids, ratings = movie2userrating[j]
            matrix = np.dot(W[user_ids].T, W[user_ids]) + reg * np.eye(K)
            vector = np.dot(ratings - b[user_ids] - c[j] - mu, W[user_ids])
            cj = np.sum(ratings - W[user_ids].dot(U[j]) - b[user_ids] - mu)

            U[j] = np.linalg.solve(matrix, vector)
            c[j] = cj / (len(ratings) + reg)
            if j % (M//10) == 0:
                print(f'j: {j}, M: {M}')
        except KeyError:
            pass
        
    print(f'updated U and c: {datetime.now() - t0}')
    print(f'epoch duration: {datetime.now() - epoch_start}')


    # store train loss
    t0 = datetime.now()
    train_losses.append(get_loss(movie2userrating))

    # store test loss
    test_losses.append(get_loss(movie2userrating_test))
    print('calculate cost:', datetime.now() - t0)
    print('train loss:', train_losses[-1])
    print('test_loss:', test_losses[-1])

print('train losses:', train_losses)
print('test_losses:', test_losses)

plt.plot(train_losses, label = 'train loss')
plt.plot(test_losses, label = 'test loss')
plt.legend()
plt.show()