import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split


path = os.path.abspath('../data')
filename = os.path.join(path,"ml-100k/u.data")
List = []
columns = ['user_id', 'movie_id', 'rating', 'timestamp']

with open(filename, 'r') as f:
    data = f.read()
    data = data.split("\n")
    for li in data:
        list_temp = li.split("\t")
        List.append(list_temp)

df = pd.DataFrame(List, columns=columns)
df.drop('timestamp', axis=1, inplace=True)
train_data, test_data  = train_test_split(df, test_size=0.3)

movies = df['movie_id'].unique()
users = df['user_id'].unique()
no_movies = len(movies)
no_users = len(users)

print(no_movies, no_users)

movie_map = {}
users_map = {}

movie = {}

filename = os.path.join(path,"ml-100k/u.item")

with open(filename, 'r') as f:
    data = f.read()
    data = data.split("\n")
    for li in data:
        l_temp = li.split("|")
        print(l_temp)
        if len(l_temp) > 1:
            movie[l_temp[0]] = l_temp[1]


print(movie)

''' creating a index mapping for users and movies '''

for k, v in enumerate(movies):
    movie_map[v] = k

for k, v in enumerate(users):
    users_map[v] = k

''' creating two dimensional utility matrix
    rows: users
    columns: movies
'''

utility_mat = np.zeros((943,1683))

print(df.shape)
print(utility_mat.shape)

print(len(df))
a = 0

for index, row in df.iterrows():
    a += 1
    #print(a, df.shape[0])
    if a == len(df) - 1:
        break
    utility_mat[users_map[row['user_id']]
                ][movie_map[row['movie_id']]] = int(row['rating'])

print(utility_mat)

''' persistent storage for the utility matrix and other data '''

file_handler = open("utility", 'wb+')
pickle.dump(utility_mat, file_handler)

file_handler = open("users_map", 'wb+')
pickle.dump(users_map, file_handler)

file_handler = open("movie_map", 'wb+')
pickle.dump(movie_map, file_handler)

file_handler = open("test", 'wb+')
pickle.dump(test_data, file_handler)

file_handler = open("train", 'wb+')
pickle.dump(train_data, file_handler)

file_handler = open("movie", 'wb+')
pickle.dump(movie, file_handler)