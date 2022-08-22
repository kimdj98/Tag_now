# DataScience

파일구조

modeling/<br>
ㄴdata_files/<br>
     &emsp;ㄴall_data.csv<br>
ㄴnon_personalized_model/<br>
     &emsp;ㄴbayesian_rating.py<br>
     &emsp;ㄴget_tags.py<br>
     &emsp;ㄴtag_based_similarity.py<br>
     
<img src="https://user-images.githubusercontent.com/81472155/185895707-0621c650-6d6a-44d8-85bb-61e3cd2a808e.png"  width="25%">


tag_based_similarity: 위 사진에서 이 식당을 좋아한 사용자들이 찾아본 식당 부분을 구현하기 위한 알고리즘을 작성한 코드<br>
유저 데이터가 없으므로 이 식당을 좋아한 사용자들이 찾아본 식당 -> 이 식당과 유사한 식당으로 수정

알고리즘 설명: 수기로 태그 표시한거에서 두개의 음식점이 있을 때<br>
r1 = [0, 0, 1, 0, 1, 1]<br>
r2 = [0, 1, 1, 0, 0, 1]

### 태그 마다의 가중치<br>
w = [1, 1, 1, 0.1, 1, 0] 

### r1과 r2의 태그 사이의 관계<br>
and= [0, 0, 1, 0, 0, 1] # r1 and r2<br>
xor = [0, 1, 0, 0, 1, 0] # r1 xor r2<br>
nor = [1, 0, 0, 1, 0, 0] # r1 nor r2

similarity = (1) * and + (-0.5) * xor + (0.1) * nor <br>
##### (1, -0.5, 0.1)은 and xor nor의 음식점 태그 사이의 관계에 관한 가중치

<img src="https://user-images.githubusercontent.com/81472155/185897203-28d6f68a-f135-4527-93be-4dbfb8029f5d.png" width="25%">

get_tags.py -> get_similar_restaurants_from_tag:<br>
get_similar_restaurants_from_tag(df, restaurant_name, tag=None) 을 치면 유사도가 높은 음식점 순으로 나오는 함수, '이 식당과 유사한 식당'란에 들어갈 음식점 리스트 반환<br>
bayesian_ranking.py: 위의 화면에서 태그 눌렀을 때 나오는 음식점을 리스트 형태로 반환하기 위한 알고리즘을 구현한 코드

알고리즘 설명:
클릭 수에 따라 변하는 확률 분포(beta 분포)를 가지고 클릭할 확률을 랜덤추출해서 확률이 높은 음식점 순서로 반환. 아래 그림에서와 같이 클릭 데이터가 많을수록(A, E) 확률의 분포가 하나의 점에 모이게 된다. 따라서 데이터가 쌓이면 더 정확하게 확률을 계산할 수 있다.

<img src="https://user-images.githubusercontent.com/81472155/185896422-232c29f1-39bd-4e55-a492-f3ba77e176c4.png" width="50%">


![image](https://user-images.githubusercontent.com/81472155/185896456-c4e50139-c9a4-4e3a-8515-66ca9702ffdd.png)



get_tags.py -> get_tag_main:
get_tag_name(df, tag='None')을 치면 클릭 수를 바탕으로 bayesian ranking을 시행, 각 음식점의 확률을 추출해서 확률이 높은 순으로 음식점 리스트를 반환
(유의사항* 매번 다른 음식점 순서로 리스트 반환)
