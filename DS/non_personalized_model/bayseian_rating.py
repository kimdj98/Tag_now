import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta
# link들의 순서를 bayseian ranking을 사용해서 가져오는 코드
# database에서 어떤 restaurants data를 가져오냐에 따라 다른 결과값 생성
# 태그에 따라 다른 결과값 추출
# click_count log_data 필요!

###############################################################
###### 데이터베이스 연결해서 restaurant, click_count) 가져오기 ######
###############################################################

pass

# restaurants의 구조: [restaurant1, restaurant2, restaurant3, ... ]
# restaurant의 구조: [restaurant, click_count]
# restaurants = []


# links 에 link들을 추가
pass


################## click_count를 별점개수로 대체해서 실험 ######################
# u_cols = ['restaurant_name', 'ratings', 'ratings_num', 'address', 'address_number', 'tags', 'tag1', 'tag2', 'tag3', 'tag4', 'tag5']

# df = pd.read_csv('./data_files/sinchon_data.csv', names=u_cols)
# df = df[['restaurant_name', 'ratings', 'ratings_num']]

# # get_restaurant: restaurant_vector dictionary pairs
# restaurant_clickcounts = []
# def get_name_clickcount_dict(row):
#     restaurant_clickcounts.append((row[0], np.int(row[2])))
#     restaurants.append(row[0])
# df.apply(get_name_clickcount_dict, axis = 1)

###########################################################################

# clickcount를 probability로 바꿈
# total_clickcount = np.sum([restaurant_clickcount[1] for restaurant_clickcount in restaurant_clickcounts])

# link의 구조: [restaurant, probability]
# probabilities = [restaurant_clickcount[1] / total_clickcount for restaurant_clickcount in restaurant_clickcounts] # restaurant[1]: click_count of restaurant

# Link object: link마다 sampling을 위해 사용!
class Link:
    def __init__(self, a, b):
        # self.p = p
        self.a = a + 1
        self.b = b + 1
        # self.N = 1

    # def expect(self):
    #     return np.random.random() < self.p

    def sample(self):
        return np.random.beta(self.a, self.b)

    # def update(self, x):
    #     self.a += x
    #     self.b += 1 - x
    #     self.N += 1


def bayesian_rank(df):
    # ------------------------------ data preprocessing for bayesian_rating starts------------------------------------ #
    df = df[['name', 'rate', 'rate_num']]
    restaurants = []

    # get_restaurant: restaurant_vector dictionary pairs
    restaurant_clickcounts = []
    def get_name_clickcount_dict(row):
        restaurant_clickcounts.append((row[0], np.int(row[2])))
        restaurants.append(row[0])
    df.apply(get_name_clickcount_dict, axis = 1)
    # ------------------------------ data preprocessing for bayesian_rating ends-------------------------------------- #

    # click_count로 probabilities 행렬을 만듬(frequent ratio)
    total_clickcount = sum([clickcount[1] for clickcount in restaurant_clickcounts])
    click_noclick_count = [(clickcount[1], total_clickcount - clickcount[1]) for clickcount in restaurant_clickcounts] # (click, noClick)
    # click_through_rate에 맞춰 link 객체 생성
    links = [Link(a,b) for (a,b) in click_noclick_count]
    results = []
    for link in links:
        # 각 링크의 확률을 표본추출
        link_prob = link.sample()
        # results list에 추가하고
        results.append(link_prob)
    
    args = np.argsort(results)
    
    # 각 링크의 뽑힌 확률이 높은 순서[args]에 맞추어서 return
    return np.array(restaurants)[args.astype(np.int32)]

# print(bayesian_rank(restaurants,)[:5])