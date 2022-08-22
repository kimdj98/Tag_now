from get_tags import get_tag_main
import pandas as pd
from flask import Flask

app = Flask(__name__)


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

@app.route('/get_tag/<tag>', methods = ['get'])
def get_main_tag(tag):
    rest_list = list(get_tag_main(df, tag=tag))
    main = {}
    for i in range(len(rest_list)):
        main[i] = rest_list[i]
    return main

if __name__ == '__main__':
    app.run(debug=True)