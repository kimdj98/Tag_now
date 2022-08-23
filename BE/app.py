from flask import Flask, redirect, url_for, render_template

# mongodb - atlas connection start
from pymongo import MongoClient
import os
from dotenv import load_dotenv

import numpy as np
import pandas as pd
from scipy.stats import beta

from operator import itemgetter

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

import numpy as np
import pandas as pd

weight_tag =    [
                1,      # 고기
                1,      # 한식
                1,      # 중식
                1,      # 양식
                1,      # 일식
                1,      # 카페
                0.1,    # 친구랑
                1,      # 술자리
                0.5,    # 밥약
                1,      # 데이트
                1,      # 디저트
                0.5,    # 혼밥
                1,      # 분위기 있는
                0.8,    # 가성비
                0.1,    # 깨끗한
                1,      # 소개팅
                0.3,    # 뒷풀이
                0       # 예약가능여부
                ]

weight_tag = np.array(weight_tag)

def similarity(restaurant2tag, r1, r2):
    '''
    calculates the similarity of 2 restaurants based on the handwritten dictionaryt data (restaurant2tag)
    '''
    r1_tags = restaurant2tag[r1]
    r2_tags = restaurant2tag[r2]
    and_array = np.logical_and(r1_tags, r2_tags)
    xor_array = np.logical_xor(r1_tags, r2_tags)
    nor_array = np.logical_not(np.logical_or(r1_tags, r2_tags))
    weighted_tag = (1) * weight_tag * and_array + (-0.5) * weight_tag * xor_array + (0.2) * weight_tag * nor_array # (1), (-0.5), (0.2) are the weights for differentiating the and, xor, nor
    
    index = 0

    return np.sum(weighted_tag)

def get_similarities(df, r): # main function of this program
    '''
    get similarities of all restaurant to r
    returns as dictionary type
    '''
    restaurant2tag = {}
    def get_name_tag_dict(row):
        restaurant2tag[row[0]] = [row[i] for i in range(3,21)]
    df.apply(get_name_tag_dict, axis = 1)

    restaurants = restaurant2tag.keys()

    similarities = {}
    for restaurant in restaurants:
        if restaurant == r: # do not evaluate similarities for same restaurant r
            pass
        else:
            similarities[restaurant] = similarity(restaurant2tag, r, restaurant)
    sorted_similarities = {k: v for k, v in sorted(similarities.items(), key=lambda item: item[1], reverse = True)}
    return sorted_similarities

load_dotenv()

USER_ID = os.getenv('USER_ID')
USER_PASSWORD = os.getenv('USER_PASSWORD')

mongodb_URI=f"mongodb+srv://{USER_ID}:{USER_PASSWORD}@tagnow.y6k5mgq.mongodb.net/Place?retryWrites=true&w=majority"
client = MongoClient(mongodb_URI)

db = client.Place
collection_a = db.All
collection_s = db.Sinchon
collection_y = db.Yeonnam
app = Flask(__name__)

#추천 알고리즘
# loading data
df = pd.read_csv('all_data.csv', header = 0)

# shrink data
df = df[['name','rate', 'rate_num',
          '고기', '한식', '중식', '양식','일식','카페', '친구랑', '술자리','밥약','데이트',
          '디저트','혼밥','분위기 있는','가성비','깨끗한','소개팅','뒷풀이', '예약가능여부']]

# select the parts which tags are implemented
df1 = df.iloc[0:338]
df2 = df.iloc[463:680]
df = pd.concat([df1, df2])

# get_tag_main uses bayesian ratings using clickcount
def get_tag_main(df, tag=None):
    if tag:
        df = df.loc[df[tag]==1] # shrink dataframe to the corresponding tag
    return bayesian_rank(df)

# get_similar_restaurants_from_tag uses tag_based_similarity using handwritten tags
def get_similar_restaurants_from_tag(df, r, tag = ''):
    if tag:
        df = df.loc[df[tag] == 1] # shrink dataframe to the corresponding tag
    return sorted(list(get_similarities(df, r).items()), key = itemgetter(1), reverse=True)[:20]

@app.route('/', methods = ['GET'])
def main():
    return "Home Page"

@app.route('/all', methods = ['GET'])
def give_all():
    list = []
    result= collection_a.find()
    idx = 0
    for i in result:
        list.append({"name": i["name"], "address" : i["address"], "r_n_address" : i["r_n_address"], "rate": i["rate"], "rate_num": i["rate_num"],\
                   "tag_1": i["tag_1"], "tag_2": i["tag_2"], "tag_3": i["tag_3"], "tag_4": i["tag_4"], "tag_5": i["tag_5"], \
                   "tag_6": i["tag_6"], "tag_7": i["tag_7"], "tag_8": i["tag_8"], "tag_9": i["tag_9"], "tag_10": i["tag_10"], \
                   "tag_11": i["tag_11"], "tag_12": i["tag_12"], "tag_13": i["tag_13"], "tag_14": i["tag_14"], "tag_15": i["tag_15"], \
                   "tag_16": i["tag_16"], "tag_17": i["tag_17"], "tag_18": i["tag_18"], "tag_19": i["tag_19"], "tag_20": i["tag_20"], "idx" : idx, "image_uri": i["image_uri"][1:-1]})
        idx += 1
    return {"restaurant_all": list}

# @app.route('/sinchon', methods = ['GET'])
# def give_sinchon():
#     list = []
#     result = collection_s.find()
#     idx = 0
#     for i in result:
#         list.append({"name": i["name"], "address" : i["address"], "r_n_address" : i["r_n_address"], "rate": i["rate"], "rate_num": i["rate_num"],\
#                    "tag_1": i["tag_1"], "tag_2": i["tag_2"], "tag_3": i["tag_3"], "tag_4": i["tag_4"], "tag_5": i["tag_5"], \
#                    "tag_6": i["tag_6"], "tag_7": i["tag_7"], "tag_8": i["tag_8"], "tag_9": i["tag_9"], "tag_10": i["tag_10"], \
#                    "tag_11": i["tag_11"], "tag_12": i["tag_12"], "tag_13": i["tag_13"], "tag_14": i["tag_14"], "tag_15": i["tag_15"], \
#                    "tag_16": i["tag_16"], "tag_17": i["tag_17"], "tag_18": i["tag_18"], "tag_19": i["tag_19"], "tag_20": i["tag_20"], "idx" : idx})
#         idx += 1
#     return {"restaurant_all": list}

# @app.route('/yeonnam', methods = ['GET'])
# def give_yeonnam():
#     list = []
#     result = collection_y.find()
#     idx = 0
#     for i in result:
#         list.append({"name": i["name"], "address" : i["address"], "r_n_address" : i["r_n_address"], "rate": i["rate"], "rate_num": i["rate_num"],\
#                    "tag_1": i["tag_1"], "tag_2": i["tag_2"], "tag_3": i["tag_3"], "tag_4": i["tag_4"], "tag_5": i["tag_5"], \
#                    "tag_6": i["tag_6"], "tag_7": i["tag_7"], "tag_8": i["tag_8"], "tag_9": i["tag_9"], "tag_10": i["tag_10"], \
#                    "tag_11": i["tag_11"], "tag_12": i["tag_12"], "tag_13": i["tag_13"], "tag_14": i["tag_14"], "tag_15": i["tag_15"], \
#                    "tag_16": i["tag_16"], "tag_17": i["tag_17"], "tag_18": i["tag_18"], "tag_19": i["tag_19"], "tag_20": i["tag_20"], "idx" : idx})
#         idx += 1
#     return {"restaurant_all": list}

# @app.route('/sinchon_rate', methods = ['GET'])
# def give_rate_sinchon():
#     rates = {}
#     result_1 = collection_s.find()
#     for i in result_1:
#         rates[i["name"]]= [i["rate"], i["rate_num"]] 
#     return {"rates": rates}


# @app.route('/yeonnam_rate', methods = ['GET'])
# def give_rate_yeonnam():
#     rates = {}
#     result_1 = collection_y.find()
#     for i in result_1:
#         rates[i["name"]]= [i["rate"], i["rate_num"]] 
#     return {"rates": rates}


@app.route('/get_tag/<tags>', methods = ['GET'])
def give_tag_a(tags):
    result = get_tag_main(df, tag=tags)
    list = []
    idx = 0
    for j in result:
        i = collection_a.find_one({"name" : j})
        list.append({"name": i["name"], "address" : i["address"], "r_n_address" : i["r_n_address"], "rate": i["rate"], "rate_num": i["rate_num"],\
                   "tag_1": i["tag_1"], "tag_2": i["tag_2"], "tag_3": i["tag_3"], "tag_4": i["tag_4"], "tag_5": i["tag_5"], \
                   "tag_6": i["tag_6"], "tag_7": i["tag_7"], "tag_8": i["tag_8"], "tag_9": i["tag_9"], "tag_10": i["tag_10"], \
                   "tag_11": i["tag_11"], "tag_12": i["tag_12"], "tag_13": i["tag_13"], "tag_14": i["tag_14"], "tag_15": i["tag_15"], \
                   "tag_16": i["tag_16"], "tag_17": i["tag_17"], "tag_18": i["tag_18"], "tag_19": i["tag_19"], "tag_20": i["tag_20"], "idx" : idx, "image_uri": i["image_uri"][1:-1]})
        idx += 1
    return {"tag_search" : list}


@app.route('/get_tag_sinchon/<tags>', methods = ['GET'])
def give_tag_s(tags):
    result = get_tag_main(df=df1, tag=tags)
    list = []
    idx = 0
    for j in result:
        i = collection_a.find_one({"name" : j})
        list.append({"name": i["name"], "address" : i["address"], "r_n_address" : i["r_n_address"], "rate": i["rate"], "rate_num": i["rate_num"],\
                   "tag_1": i["tag_1"], "tag_2": i["tag_2"], "tag_3": i["tag_3"], "tag_4": i["tag_4"], "tag_5": i["tag_5"], \
                   "tag_6": i["tag_6"], "tag_7": i["tag_7"], "tag_8": i["tag_8"], "tag_9": i["tag_9"], "tag_10": i["tag_10"], \
                   "tag_11": i["tag_11"], "tag_12": i["tag_12"], "tag_13": i["tag_13"], "tag_14": i["tag_14"], "tag_15": i["tag_15"], \
                   "tag_16": i["tag_16"], "tag_17": i["tag_17"], "tag_18": i["tag_18"], "tag_19": i["tag_19"], "tag_20": i["tag_20"], "idx" : idx, "image_uri": i["image_uri"][1:-1]})
        idx += 1
    return {"tag_search" : list}


@app.route('/get_tag_yeonnam/<tags>', methods = ['GET'])
def give_tag_y(tags):
    result = get_tag_main(df=df2, tag=tags)
    list = []
    idx = 0
    for j in result:
        i = collection_a.find_one({"name" : j})
        list.append({"name": i["name"], "address" : i["address"], "r_n_address" : i["r_n_address"], "rate": i["rate"], "rate_num": i["rate_num"],\
                   "tag_1": i["tag_1"], "tag_2": i["tag_2"], "tag_3": i["tag_3"], "tag_4": i["tag_4"], "tag_5": i["tag_5"], \
                   "tag_6": i["tag_6"], "tag_7": i["tag_7"], "tag_8": i["tag_8"], "tag_9": i["tag_9"], "tag_10": i["tag_10"], \
                   "tag_11": i["tag_11"], "tag_12": i["tag_12"], "tag_13": i["tag_13"], "tag_14": i["tag_14"], "tag_15": i["tag_15"], \
                   "tag_16": i["tag_16"], "tag_17": i["tag_17"], "tag_18": i["tag_18"], "tag_19": i["tag_19"], "tag_20": i["tag_20"], "idx" : idx, "image_uri": i["image_uri"][1:-1]})
        idx += 1
    return {"tag_search" : list}
    
@app.route('/<name>', methods = ['GET'])
def give_place(name):
    list = []
    idx = 0
    result = collection_a.find({"name" : {"$regex" : f".*{name}.*"}})  
    for i in result:
        list.append({"name": i["name"], "address" : i["address"], "r_n_address" : i["r_n_address"], "rate": i["rate"], "rate_num": i["rate_num"],\
                   "tag_1": i["tag_1"], "tag_2": i["tag_2"], "tag_3": i["tag_3"], "tag_4": i["tag_4"], "tag_5": i["tag_5"], \
                   "tag_6": i["tag_6"], "tag_7": i["tag_7"], "tag_8": i["tag_8"], "tag_9": i["tag_9"], "tag_10": i["tag_10"], \
                   "tag_11": i["tag_11"], "tag_12": i["tag_12"], "tag_13": i["tag_13"], "tag_14": i["tag_14"], "tag_15": i["tag_15"], \
                   "tag_16": i["tag_16"], "tag_17": i["tag_17"], "tag_18": i["tag_18"], "tag_19": i["tag_19"], "tag_20": i["tag_20"], "idx" : idx, "image_uri": i["image_uri"][1:-1]})
        idx += 1
    return {"restaurants": list}

@app.route('/get_similar_sinchon/<name>', methods = ['GET'])
def give_similar_sinchon(name):
    
    result = get_similar_restaurants_from_tag(df1, name)
    list = []
    idx = 0
    for j in result:
        i = collection_a.find_one({"name" : j[0]})
        list.append({"name": i["name"], "address" : i["address"], "r_n_address" : i["r_n_address"], "rate": i["rate"], "rate_num": i["rate_num"],\
                   "tag_1": i["tag_1"], "tag_2": i["tag_2"], "tag_3": i["tag_3"], "tag_4": i["tag_4"], "tag_5": i["tag_5"], \
                   "tag_6": i["tag_6"], "tag_7": i["tag_7"], "tag_8": i["tag_8"], "tag_9": i["tag_9"], "tag_10": i["tag_10"], \
                   "tag_11": i["tag_11"], "tag_12": i["tag_12"], "tag_13": i["tag_13"], "tag_14": i["tag_14"], "tag_15": i["tag_15"], \
                   "tag_16": i["tag_16"], "tag_17": i["tag_17"], "tag_18": i["tag_18"], "tag_19": i["tag_19"], "tag_20": i["tag_20"], "idx" : idx, "image_uri": i["image_uri"][1:-1], \
                   "similarity": j[1]})
        idx += 1
    return {"similar" : list}

@app.route('/get_similar_yeonnam/<name>', methods = ['GET'])
def give_similar_yeonnam(name):
    
    result = get_similar_restaurants_from_tag(df2, name)
    list = []
    idx = 0
    for j in result:
        i = collection_a.find_one({"name" : j[0]})
        list.append({"name": i["name"], "address" : i["address"], "r_n_address" : i["r_n_address"], "rate": i["rate"], "rate_num": i["rate_num"],\
                   "tag_1": i["tag_1"], "tag_2": i["tag_2"], "tag_3": i["tag_3"], "tag_4": i["tag_4"], "tag_5": i["tag_5"], \
                   "tag_6": i["tag_6"], "tag_7": i["tag_7"], "tag_8": i["tag_8"], "tag_9": i["tag_9"], "tag_10": i["tag_10"], \
                   "tag_11": i["tag_11"], "tag_12": i["tag_12"], "tag_13": i["tag_13"], "tag_14": i["tag_14"], "tag_15": i["tag_15"], \
                   "tag_16": i["tag_16"], "tag_17": i["tag_17"], "tag_18": i["tag_18"], "tag_19": i["tag_19"], "tag_20": i["tag_20"], "idx" : idx, "image_uri": i["image_uri"][1:-1], \
                   "similarity": j[1]})
        idx += 1
    return {"similar" : list}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port = 5000, debug=True)