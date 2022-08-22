import pandas as pd

df = pd.read_csv('/home/kimdj/my_folder/large_files/moviedata/rating.csv')

# make the user ids go from 0...N-1
df.userId = df.userId - 1

# create a mapping for movie ids
unique_movie_ids = set(df.movieId.values)

# create a mapping for movie ids
unique_movie_ids = set(df.movieId.values)
movie2idx = {}
count = 0
for movie_id in unique_movie_ids:
    movie2idx[movie_id] = count
    count += 1

# add them to the dataframe
# takes a while
df['movie_idx'] = df.apply(lambda row: movie2idx[row.movieId], axis = 1)

df = df.drop(columns = ['timestamp'])

df.to_csv('/home/kimdj/my_folder/large_files/moviedata/edited_rating.csv')

# for debugging
print(df)
