import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime
from sortedcontainers import SortedList

# load in the data
import os
if not os.path.exists('user2movie.json') or \
   not os.path.exists('movie2user.json') or \
   not os.path.exists('usermovie2rating.json') or \
   not os.path.exists('usermovie2rating_test.json'):
   import preprocess2dict

# read user2movie.json
with open('user2movie.json', 'rb') as f:
    user2movie = pickle.load(f)

with open('movie2user.json', 'rb') as f:
    movie2user = pickle.load(f)

with open('usermovie2rating.json', 'rb') as f:
    usermovie2rating = pickle.load(f)

with open('usermovie2rating_test.json', 'rb') as f:
    usermovie2rating_test = pickle.load(f)

df = pd.read_csv('/home/kimdj/my_folder/large_files/very_small_rating.csv',
                index_col = 0)
df = df.reset_index(drop = True)
# for debugging purpose
# print(df)
# print(list(user2movie.items())[:2])
# print(list(movie2user.items())[:2])
# print(list(usermovie2rating.items())[:2])
# print(list(usermovie2rating_test.items())[:2])

num_users = df.userId.max() + 1 # number of users
num_movies = df.movieId.max() + 1 # number of movies

# for debugging purpose
# print('num_users:', num_users, 'num_movies:', num_movies)

# data preprocessing ends

##################### start making dataframes and user_set for prediction ######################
# no machine-learning techniques involved (no learning process)
# user_based collaborative filtering algorithms starts
'''
user2movie_mean: code for getting average rate of user
'''
user_set = set(df.userId)
user2movie_mean = {}

for user in user_set:
    user_rate = []
    movies = user2movie[user]
    for movie in movies:
        rate_train = usermovie2rating[user, movie]
        user_rate.append(rate_train)

    user_mean = np.mean(user_rate)
    user2movie_mean[user] = user_mean

# for debugging purpose
# print(list(user2movie_mean.items()))

'''
usermovie2rating -> usermovie2dev_rating: user,movie -> rating_dev
'''
usermovie2dev_rating = {}
for user in user_set:
    movies = user2movie[user]
    for movie in movies:
        usermovie2dev_rating[user, movie] = usermovie2rating[user, movie] - user2movie_mean[user]

# for debugging purpose
# print('usermovie2dev_rating shows (user, item) -> rating in list form')
# print(list(usermovie2dev_rating.items())[:5])    

'''
user2user_set: all combination of user, user collected as python set
user2user_similarity: similarity matrix from user2user
'''

# for debugging purpose

K = 25 # number of neighbors we'd like to consider
limit = 5 # number of common movies users must have in common in order to consider
neighbors = []
averages = []
deviations = []

max_userId = df.userId.max()
max_movie_idx = df.movie_idx.max()

# uses 2 for loops but it has relatively small common movies so it should be fine in terms of bigO of O(n^2)
def predict(userid, movieid):
    userid = int(userid)
    movieid = int(movieid)
    user_set_temp = user_set.copy() # copy user_set to avoid removing problem
    user_set_temp.remove(userid)
    count = 0
    weights = {}
    for user in user_set:
        common_movie_idx = list(set(user2movie[userid]) & set(user2movie[user]))
        if len(common_movie_idx) < 5:
            user_set_temp.remove(user)
        else:
            count += 1
            # weight 구하는 과정
            numerator = 0
            denominator1 = 0
            denominator2 = 0
            for common_movie in common_movie_idx:
                numerator += usermovie2dev_rating[userid, common_movie]*usermovie2dev_rating[user, common_movie]
                denominator1 += usermovie2dev_rating[userid, common_movie]**2
                denominator2 += usermovie2dev_rating[user, common_movie]**2
            weight = numerator / (np.sqrt(denominator1) * np.sqrt(denominator2))
            weights[user] = weight
    
    for user in user_set:
        numerator = 0
        denominator = 0
        if (user,movie) in usermovie2dev_rating:
            numerator += weights[user]*usermovie2dev_rating[user,movie]
            denominator += np.abs(weights[user])
    if denominator != 0:
        weighted_dev = numerator/denominator
    else:
        weighted_dev = 0 # 겹치는 영화가 없을 시 해당 user의 평균으로 예측함
    return user2movie_mean[userid] + weighted_dev


def main():
    userid = input('0 ~ 9999 사이의 userId를 입력해 주세요: ')

    movieid = input('0 ~ 1999 사이의 movie_idx를 입력해 주세요: ')

    if (userid, movieid) in usermovie2rating:
        print('data exists!')
        return usermovie2rating[userid, movieid]
    elif (userid, movieid) in usermovie2rating_test:
        print('data exists!')
        return usermovie2rating_test[userid, movieid]
    else:
        return predict(userid, movieid)

def rmse(p, q):
    p = np.array(p)
    q = np.array(q)
    return np.sqrt(np.sum((p-q)**2))

if __name__ == '__main__':
    # user_id 와 movie_idx를 terminal에서 입력받아서 predict값을 받는 코드 
    # predicted = main()
    # print(predicted)
    predicted = {}
    for (user, movie) in usermovie2rating_test.keys():
        print(user,movie)
        predicted[user, movie] = predict(user, movie)
    predicted_ratings = predicted.values()
    test_ratings = usermovie2rating_test.values()
    print(rmse(predicted_ratings, test_ratings))