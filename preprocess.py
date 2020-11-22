import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

path = os.path.abspath("./data")
file1 = os.path.join(path,"ml-100k/u.data")
tempo_list = []
columns = ['user_id', 'movie_id', 'rating', 'timestamp']

with open(file1, 'r') as f:
    data = f.read()
    data = data.split("\n")
    for li in data:
        list_temp = li.split("\t")
        tempo_list.append(list_temp)
rating_df = pd.DataFrame(tempo_list, columns=columns)
rating_df.drop('timestamp', axis=1, inplace=True)

print(rating_df)

''' 
    creating a index mapping for users and movies 

    creating two dimensional utility matrix
    rows: users
    columns: movies
'''
utility_mat = np.zeros((943, 1683))

print(rating_df.shape)
print(utility_mat.shape)

a = 0

for index, row in rating_df.iterrows():
    a += 1
    #print(a, "/", rating_df.shape[0])
    if a == len(rating_df) - 1:
        break
    utility_mat[int(row['user_id'])-1][int(row['movie_id'])-1] = int(row['rating'])

#print(utility_mat)
#print(utility_mat.shape)