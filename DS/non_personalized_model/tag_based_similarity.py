## model for non_personalized model ##

# import modules
import numpy as np
import pandas as pd

#--------------------moved inside the function------------------
# # import data
# df = pd.read_csv('../data_files/all_data.csv', header = 0)

# # shrink data
# df = df[['name','rate', 'rate_num',
#           '고기', '한식', '중식', '양식','일식','카페', '친구랑', '술자리','밥약','데이트',
#           '디저트','혼밥','분위기 있는','가성비','깨끗한','소개팅','뒷풀이', '예약가능여부']]

# # select the parts which tags are implemented
# df1 = df.iloc[0:249]
# df2 = df.iloc[480:629]
# df = pd.concat([df1, df2])

# # for debugging purpose
# # print(df.describe)
# # print(df.columns)


# # get {restaurant:tag} dictionary pairs
# restaurant2tag = {}
# def get_name_tag_dict(row):
#     restaurant2tag[row[0]] = [row[i] for i in range(3,21)]
# df.apply(get_name_tag_dict, axis = 1)

# restaurants = restaurant2tag.keys()


# weight_tag: 태그마다 가중치를 다르게 하기 위해 설정(수치는 수기로 조정)
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

# testing if code works(debugging purpose)

# similarities = get_similarities('라오샹하이')
# similar_to = list(similarities.items())[0:30:4]
# print(similar_to)

'''
음식점 이름,
주소,
태그 수기로 한 거 1값
'''