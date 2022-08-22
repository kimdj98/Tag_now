## 이 식당을 좋아한 사람들이 좋아한 식당 리스트 반환 코드 (특정 태그내에서) ##
from bayseian_rating import bayesian_rank
from tag_based_similarity import get_similarities

import pandas as pd

from operator import itemgetter

# loading data
df = pd.read_csv('../data_files/all_data.csv', header = 0)

# shrink data
df = df[['name','rate', 'rate_num',
          '고기', '한식', '중식', '양식','일식','카페', '친구랑', '술자리','밥약','데이트',
          '디저트','혼밥','분위기 있는','가성비','깨끗한','소개팅','뒷풀이', '예약가능여부']]

# select the parts which tags are implemented
df1 = df.iloc[0:249]
df2 = df.iloc[480:629]
df = pd.concat([df1, df2])

# get_tag_main uses bayesian ratings using clickcount
def get_tag_main(df, tag=None):
    if tag:
        df = df.loc[df[tag]==1] # shrink dataframe to the corresponding tag
    return bayesian_rank(df)

# for debugging
# print(get_tag_main(df))
# output: ['서대문양꼬치' '헬로케이크' '이찌방이야기' '국제통닭 신촌점' '신촌영양센터' '판자집' '원조감자탕' '신촌양꼬치' '앤스' ... 


# get_similar_restaurants_from_tag uses tag_based_similarity using handwritten tags
def get_similar_restaurants_from_tag(df, r, tag = ''):
    if tag:
        df = df.loc[df[tag] == 1] # shrink dataframe to the corresponding tag
    return sorted(list(get_similarities(df, r).items()), key = itemgetter(1), reverse=True)
# for debugging
# print(get_similar_restaurants_from_tag(df,'길상양꼬치','고기'))
# output: [('장작집', 3.5599999999999996), ('신촌양꼬치', 3.1799999999999997), ('아웃닭 이대점', 3.1799999999999997), ('대현매운족발 본점', 3.1799999999999997), ('대구삼겹살', 3.1799999999999997)]

