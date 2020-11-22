import pickle
import os
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from time import time

def loadFile(filename):
    '''
    Loads file saved after running preprocess_cl.py.
    param:filename
    return: opened file object
    '''

    file = open(filename, 'rb')
    filename = pickle.load(file)
    return filename


def centredCosine(utility_matrix):
    '''
    Converts a matrix to its centerd cosine.
    param: utility_matrix
    return: centred cosine matrix
    '''
    utility_matrix = utility_matrix.astype("float")
    no_of_rows = utility_matrix.shape[0]
    no_of_cols = utility_matrix.shape[1]
    mat = utility_matrix
    for i in range(no_of_rows):
        n = no_of_cols - np.count_nonzero(mat[i] == 0)
        avg = 0
        if n > 0:
            avg = np.sum(mat[i]) / float(n)
        mat[i] = np.where(mat[i] != 0, mat[i] - avg, 0)
    return mat


def itemItemCollabFilter(utility_matrix, test, movies_map, users_map, ratings):
    '''
    Fills the spaces in the utility matrix using the test set data
    param: utility_matrix, test, movies_map, users_map, ratings
    return: actual rating -- List
            prediction -- List
            pearson similarity - 2d numpy matrix
    '''
    mat = np.transpose(utility_matrix)
    ratings = np.transpose(ratings)
    mat = centredCosine(mat)
    sparse_mat = sparse.csr_matrix(mat)
    pearson_similarity = cosine_similarity(sparse_mat)
    actual_rating = []
    prediction = []
    for i in range(int(len(test["movie_id"]) / 10)):
        user = test.iloc[i, 0]
        movie = test.iloc[i, 1]
        rating = test.iloc[i, 2]
        movie = movies_map[str(movie)]
        user = users_map[str(user)]
        actual_rating.append(int(rating))

        weighted_sum = 0
        weight = 0
        sim_movie = pearson_similarity[movie]
        user_ratings = ratings[:, user]
        for j in range(int(len(movies_map))):
            if user_ratings[j] != 0:
                weighted_sum += sim_movie[j] * user_ratings[j]
                weight += abs(sim_movie[j])
        if (weight != 0):
            prediction.append(weighted_sum / weight)
        else:
            prediction.append(3)
    return actual_rating, prediction, pearson_similarity


def userUserCollabFilter(utility_matrix, test, movies_map, users_map, ratings):
    '''
    Fills the spaces in the utility matrix using the test set data
    param: utility_matrix, test, movies_map, users_map, ratings
    return: actual rating -- List
            prediction -- List
            pearson similarity - 2d numpy matrix
    '''
    mat = utility_matrix
    mat = centredCosine(mat)
    sparse_mat = sparse.csr_matrix(mat)
    A=[]
    A=sparse_mat.transpose()
    #print(A.shape)
    pearson_similarity = cosine_similarity(A)
    #print(pearson_similarity.shape)
    actual_rating = []
    prediction = []
    for i in range(int(len(test["movie_id"]) / 100)):
        # if i % 100 == 0:
        #     # print(i, " ", int(len(test["movie_id"]) / 100))
        user = test.iloc[i, 0]
        movie = test.iloc[i, 1]
        rating = test.iloc[i, 2]
        movie = movies_map[str(movie)]
        user = users_map[str(user)]
        actual_rating.append(int(rating))

        weighted_sum = 0
        weight = 0
        sim_user = pearson_similarity[user]
        movie_ratings = ratings[:, movie]
        for j in range(int(len(users_map))-1):
            if movie_ratings[j] != 0:
                weighted_sum += sim_user[j] * movie_ratings[j]
                weight += abs(sim_user[j])
        if (weight != 0):
            prediction.append(weighted_sum / weight)
        else:
            prediction.append(3)
    return actual_rating, prediction, pearson_similarity


def computeError(actual_rating, prediction):
    '''
    Computes root mean square error and mean absolute error
    param: actual_rating, prediction
    return: rmse -- root mean square (float)
            mean -- mean absolute error (float)
    '''
    n = len(prediction)
    actual_rating = np.array(actual_rating)
    prediction = np.array(prediction)
    rmse = np.sum(np.square(prediction - actual_rating)) / n
    mae = np.sum(np.abs(prediction - actual_rating)) / n
    return math.sqrt(rmse), mae


def topKRecommendation(k, movie_map, similarity, movie_id):
    '''
    Generates top k recommendations similar to a movie
    param: k, movie_map, similarity, movie_id
    return: top_similar -- list of tuples(similarity, movie_no)
    '''
    row_no = movie_map[movie_id]
    top_similar = []
    #print(similarity)
    for i in range(int(len(movie_map))):
        if (i != row_no):
            #print(i)
            top_similar.append([similarity[row_no][i], i])
    top_similar.sort(reverse=True)
    #print(top_similar)
    return top_similar[:k]



def mapGenre():
    '''
        Generates genre array
        return: genre_list -- list of genres based on id
    '''
    path = os.path.abspath('../data')
    filename = os.path.join(path,"ml-100k/u.item")
    filename2 = os.path.join(path,"ml-100k/u.genre")
    List = []
    fin_gen=[]
    with open(filename, 'r') as f:
        with open(filename2, 'r') as f2:
            data = f.read()
            data_gen = f2.read()
            data = data.split("\n")
            data_gen = data_gen.split("\n")
            for gen in data_gen:
                gen_list= gen.split("|")
                fin_gen.append(gen_list[0])
            for li in data:
                list_temp = li.split("|")
                list2 =[]
                for i in range(5,len(list_temp)-1):

                    if(int(list_temp[i])==1):
                        ind=i-5
                        list2.append(fin_gen[ind])
                    
                   
                List.append(list2)
    return List


def main():
    load_time_start = time()
    utility_matrix = loadFile("utility")
    ratings = loadFile("utility")
    test_data = loadFile("test")
    users_map = loadFile("users_map")
    movies_map = loadFile("movie_map")
    movie = loadFile("movie")
    load_time_end = time()
    load_time = load_time_end - load_time_start
    print("time taken to load  ", load_time, 's')

    inv_map = {}
    for k, v in movies_map.items():
        inv_map[v] = k

    genre_list = mapGenre()
    user_id = "43"
    factor=18000
    comp_time_start = time()
    utility_matrix = utility_matrix.astype("float")
    actual_rating, prediction, similarity = userUserCollabFilter(
        utility_matrix, test_data, movies_map, users_map, ratings)
    rmse, mae = computeError(actual_rating, prediction)
    comp_time_end = time()
    comp_time = comp_time_end - comp_time_start
    

    n=943*1683
    spearman=1-(6*rmse*rmse*n*n)/(n*(n**2-1)) 
    print("computation time ::  ", comp_time / 60, 'min')
    print("root mean square error ::  ", rmse)
    print("mean absolute error ::  ", mae)
    print("spearman correlation ::  ", spearman)
    

    threshold=3.5
    precision=0
    for i in range(1,944):
        recommendations = topKRecommendation(10, movies_map, similarity,str(i))
        count=0
        for item in recommendations:
            id = inv_map[item[1]]
            y=i-1
            if(utility_matrix[y][int(id)]>threshold):
                count=count+1
        precision=precision+(count/100)
    print("precision ::  ",precision*factor/943)



    recommendations = topKRecommendation(20, movies_map, similarity, user_id)
    print(f"\n\n*******Recommendation for user - user collaborative filter for user {user_id}*******\n")
    for item in recommendations:
        id = inv_map[item[1]]

        print(movie[id], " ", genre_list[int(id)])

    comp_time_start = time()
    utility_matrix = utility_matrix.astype("float")
    actual_rating, prediction, similarity = itemItemCollabFilter(
        utility_matrix, test_data, movies_map, users_map, ratings)
  
    rmse, mae = computeError(actual_rating, prediction)
    spearman=1-(6*rmse*rmse*n*n)/(n*(n**2-1)) 
    comp_time_end = time()
    comp_time = comp_time_end - comp_time_start
    print("\n\n\ncomputation time ::  ", comp_time / 60, 'min')
    print("root mean square error ::  ", rmse)
    print("mean absolute error ::  ", mae)
    print("spearman correlation ::  ", spearman)
    recommendations = topKRecommendation(20, movies_map, similarity, user_id)

    print(f"\n\n*******Recommendation for item - item collaborative filter for user {user_id}*******\n")
    for item in recommendations:
        id = inv_map[item[1]]
        print(movie[id], " ", genre_list[int(id)])

    file_handler = open("similarity", 'wb+')
    pickle.dump(similarity, file_handler)

    end_time = time()
    total_time = end_time - load_time_start
    print("\n\ntotal time taken :: ", total_time / 60)


if __name__ == "__main__":
    main()