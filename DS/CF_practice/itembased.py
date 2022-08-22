# item based collaborative filtering
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime
from sortedcontainers import SortedList

# load in the data
import os
if not os.path.exists('CF_practice/user2movie.json') or \
   not os.path.exists('CF_practice/movie2user.json') or \
   not os.path.exists('CF_practice/usermovie2rating.json') or \
   not os.path.exists('CF_practice/usermovie2rating_test.json'):
   import preprocess2dict

# read user2movie.json
with open('CF_practice/user2movie.json', 'rb') as f:
    user2movie = pickle.load(f)

with open('CF_practice/movie2user.json', 'rb') as f:
    movie2user = pickle.load(f)

with open('CF_practice/usermovie2rating.json', 'rb') as f:
    usermovie2rating = pickle.load(f)

with open('CF_practice/usermovie2rating_test.json', 'rb') as f:
    usermovie2rating_test = pickle.load(f)

# debugging purpose
# print(len(user2movie.values()))
# print(len(movie2user.values()))
print(set(movie2user.keys()))
N = np.max(list(user2movie.keys())) + 1
# the test set may contain movies the train set doesn't have data on
m1 = np.max(list(movie2user.keys()))
m2 = np.max([m for (u,m), r in usermovie2rating_test.items()])
M = max(m1,m2) + 1

# # just to make sure you didn't accidently loaded other files
# if M > 2000:
#     print('N=', M, 'are you sure you want to continue?')
#     print('comment out these lines if so...')
#     exit()

K = 20
limit = 5
neighbors = []
averages = []
deviations = []

for i in range(M):
    # find the K closest items to item 1
    users_i = movie2user[i]
    users_i_set = set(users_i)

    # calcuate avg and deviation
    ratings_i = { user:usermovie2rating[(user, i)] for user in users_i }
    avg_i = np.mean(list(ratings_i.values()))
    dev_i = { user:(rating - avg_i) for user, rating in ratings_i.items()}
    dev_i_values = np.array(list(dev_i.values()))
    sigma_i = np.sqrt(dev_i_values.dot(dev_i_values))

    # save these for later use
    averages.append(avg_i)
    deviations.append(dev_i)

    sl = SortedList()
    for j in range(M):
        # don't include yourself
        if j!= i:
            users_j = movie2user[j]
            users_j_set = set(users_j)
            common_users = (users_i_set & users_j_set) # intersection of movie i's user and movie j's user
            if len(common_users) > limit:
                # calculate avg and deviation
                ratings_j = { user:usermovie2rating[(user, j)] for user in users_j }
                avg_j = np.mean(list(ratings_j.values()))
                dev_j = { user:(rating - avg_j) for user, rating in ratings_j.items() }
                dev_j_values = np.array(list(dev_j.values()))
                sigma_j = np.sqrt(dev_j_values.dot(dev_j_values))

                # calculate correlation coefficient
                numerator = sum(dev_i[m]*dev_j[m] for m in common_users)
                w_ij = numerator / (sigma_i * sigma_j)

                sl.add((-w_ij, j))
                if(len(sl) > K):
                    del sl[-1]
        # store the neighbors
        neighbors.append(sl)

        # print out useful things
        if i % 1 == 0:
            print(i)

def predict(i, u):
    # calculate the weighted sum of deviations
    numerator = 0
    denominator = 0
    for neg_w, j in neighbors[i]:
        try:
            numerator += -neg_w * deviations[j][u]
            denominator += abs(neg_w)
        except KeyError:
            # neighbor may not have been rated by the same user
            # don't want to do dictionary lookup twice
            # so just throw exception
            pass
    if denominator == 0:
        prediction = averages[i]
    else:
        prediction = numerator / denominator + averages[i]
    prediction = min(5, prediction)
    prediction = max(0.5, prediction)
    return prediction



train_predictions = []
train_targets = []
for (u,m), target in usermovie2rating.items():
    # calculate the prediction for this movie and user
    prediction = predict(u,m)

    # save the prediction and target
    train_predictions.append(prediction)
    train_targets.append(target)

test_predictions = []
test_targets = []
for (u,m), target in usermovie2rating_test.items():
    # calculate the prediction for this movie and user
    prediction = predict(m,u)

    # save the prediction and target
    test_predictions.append(prediction)
    test_targets.append(target)

def mse(p,t):
    p = np.array(p)
    t = np.array(t)
    return np.mean((p-t)**2)

print('train mse:', mse(train_predictions, train_targets))
print('test mse:', mse(test_predictions, test_targets))


    
''' # my code
movie2rating_avg = {}
users = user2movie.keys()
movies = movie2user.keys()

for movie in movies:
    user_count = 0
    rate_sum = 0
    for user in movie2user[movie]:
        user_count += 1
        rate_sum += usermovie2rating[(user, movie)]
    movie2rating_avg[movie] = rate_sum / user_count

usermovie2rating_dev = {}
for movie in movies:
    for user in movie2user[movie]:
        usermovie2rating_dev[(user,movie)] = usermovie2rating[(user,movie)] - movie2rating_avg[movie]

# print(usermovie2rating_dev)

# calculate weight of movie1 and movie2 based on how similar m1 and m2 are using cosine coefficient
def weight(m1, m2):
    m1_rated_user = set(movie2user[m1])
    m2_rated_user = set(movie2user[m2])
    m1_and_m2_rated_user = m1_rated_user.intersection(m2_rated_user)
    print(m1_and_m2_rated_user)
    m1_rating_dev = []
    m2_rating_dev = []
    for user in m1_and_m2_rated_user:
        m1_rating_dev.append(usermovie2rating_dev[(user,m1)])
        m2_rating_dev.append(usermovie2rating_dev[(user,m2)])
    m1_rating_dev = np.array(m1_rating_dev)
    m2_rating_dev = np.array(m2_rating_dev)
    return np.sum(m1_rating_dev * m2_rating_dev) / (np.sqrt(m1_rating_dev**2) * np.sqrt(m2_rating_dev**2))

# predict user, movie combination using itembased collaborative filtering 
def predict(u, m):
    rating_avg = movie2rating_avg[m]
    u_rated_movies = user2movie[u]
    weights = []
    devs = []
    for movie in u_rated_movies:
        weights.append(weight(m, movie))
        devs.append(usermovie2rating_dev[(user,movie)])
    return rating_avg + np.dot(weights, devs) / np.sum(np.abs(weights))

# getting MSE for evaluating algorithm performance
def mse(u, v):
    u = np.array(u)
    v = np.array(v)
    np.mean((u-v)**2)

# score the algorithms performance using MSE
def score():
    predictions = []
    actual_ratings = []
    for user, movie in usermovie2rating.keys():
        predictions.append(predict(user, movie))
        actual_ratings.append(usermovie2rating(user,movie))
    return mse(predictions, actual_ratings)

print(list(usermovie2rating_test.keys())[:5])
print(predict(2000,15))
'''