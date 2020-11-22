from helper import svd_helper 
import numpy as np
import math
import pickle
import time
import random

from numpy import linalg as la


def precision_k(Actual, Predicted):
    """
    Function to calculatethe precision at rank k
    :param Actual: The original user-movie matrix
    :param Predicted: The reconstructed user-movie rating matrix
    :return The value of Precision
    """
    precision_list = []
    threshold = 3.5

    Actual = np.asarray(Actual, dtype=np.float32)
    k = 3
    for i in range(Actual.shape[0]):
        rating_dict = {}
        for j in range(Actual.shape[1]):
            rating_dict[j] = [Predicted[i][j], Actual[i][j]]
        
        var = {k: v for k, v in sorted(rating_dict.items(), key=lambda item: item[1], reverse=True)}
        count = 0
        rel_recom = 0
        for i in var.keys():
            if count<k:
                count += 1
                if var[i][1] > threshold:
                    rel_recom += 1

        temp = rel_recom/k
        #print(temp)
        precision_list.append(temp)

    avg_precision = np.average(precision_list)
    #print(avg_precision)

    return avg_precision

def get_rms_error(orig_matrix, reconstructed_matrix):
    """
    Function to calculate RMSE error between the 2 matrices
    :param orig_matrix: The original user-movie matrix
    :param reconstructed_matrix: The reconstructed user-movie rating matrix
    :return The rmse error
    """
    error = 0
    N = len(reconstructed_matrix)
    M = len(reconstructed_matrix[0])
    for i in range(len(reconstructed_matrix)):
        for j in range(len(orig_matrix[i])):
            error += math.pow(
                reconstructed_matrix[i][j] - orig_matrix[i][j], 2)

    return math.sqrt(error/(N*M))

def forbenius_norm(orig_matrix):
    """
    Function to calculate frobenius norm values for different rows and columns
    :param input_matrix: The matrix whose frobenius norm values are to be calculated
    :return Two tuples containing frobenius norm values for rows and columns in a pair-wise format
    """

    col_Fnorms, row_Fnorms = ([], [])

    norm_of_matrix = 0
    no_rows = len(orig_matrix)
    no_cols = len(orig_matrix[0])


    for i in range(no_rows):
        for j in range(no_cols):
            norm_of_matrix += math.pow(orig_matrix[i][j],2)

    for i in range(no_rows):
        row_sum = 0
        for j in range(no_cols):
            row_sum += math.pow(orig_matrix[i][j],2)
        row_Fnorms.append((row_sum/ norm_of_matrix, i))
    
    for i in range(no_cols):
        col_sum = 0
        for j in range(no_rows):
            col_sum += math.pow(orig_matrix[j][i],2)
        col_Fnorms.append((col_sum/ norm_of_matrix, i))

    row_Fnorms.sort(reverse= True)
    col_Fnorms.sort(reverse= True)

    return (row_Fnorms, col_Fnorms)

def process_cur(orig_matrix, r):
    """
    Function to decompose input matrix using cur decomposition into three matrices
    :param input_matrix: The matrix whose decomposition is to be calculated
    :param r: r-rank approximation for input matrix
    :return Three matrices C, U, R formed after decomposing input matrix     
    """

    row_Forb_norms, col_Forb_norms = forbenius_norm(orig_matrix)
    C = []
    R = [] 

    ## R and C getting the sampled rows and cols from the orig_matrix
    for i in range(r):
        R.append(orig_matrix[row_Forb_norms[i][1]])
        C.append(list(row[col_Forb_norms[i][1]] for row in orig_matrix ))
    
    C = np.transpose(C)

    for i in range(r):
        scale = math.sqrt(r*row_Forb_norms[i][0])
        for j in range(len(R[0])):
            R[i][j] = R[i][j] / scale

    for i in range(r):
        scale = math.sqrt(r*col_Forb_norms[i][0])
        for j in range(len(C)):
            C[j][i] = C[j][i] / scale

    ## W = intersection of sampled rows and columns
    W = []

    for i in range(r):
        temp = []
        for j in range(r):
            temp.append(orig_matrix[row_Forb_norms[i][1]][col_Forb_norms[j][1]])
        W.append(temp)

    
    W = np.asarray(W, dtype=np.float32)
    temp = svd_helper(W)
    X, sigma, Yt = temp.decompose()

    for i in range(len(sigma)):
        for j in range(len(sigma[0])):
            if sigma[i][j] != 0:
                sigma[i][j] = 1 / sigma[i][j]

    Y = np.transpose(Yt)
    Xt = np.transpose(X)
    U = np.dot(Y, np.dot(np.dot(sigma, sigma), Xt))

    return (C, U, R)

def cur_90_energy(input_matrix, r):
    """
    Function to decompose input matrix using cur decomposition into three matrices retaining 90% energy
    :param input_matrix: The matrix whose decomposition is to be calculated
    :param r: r-rank approximation for input matrix
    :return Three matrices C, U, R formed after decomposing input matrix     
    """
    row_Forb_norms, col_Forb_norms = forbenius_norm(input_matrix)
    C = []
    R = [] 
    ## R and C getting the sampled rows and cols from the input_matrix
    i=0
    j=0
    r_sum = 0
    c_sum = 0
    while i<r :
        r_sum += row_Forb_norms[j][0] 
        c_sum += col_Forb_norms[j][0]
        print(r_sum, c_sum)
        ## Checking for the threshold of 90%
        if(r_sum>=0.9 and c_sum>=0.9):
            print(j)
            R.append(input_matrix[row_Forb_norms[j][1]])
            C.append(list(row[col_Forb_norms[j][1]] for row in input_matrix ))
            i+=1

        j+=1            
    
    C = np.transpose(C)

    for i in range(r):
        scale = math.sqrt(r*row_Forb_norms[i][0])
        for j in range(len(R[0])):
            R[i][j] = R[i][j] / scale

    for i in range(r):
        scale = math.sqrt(r*col_Forb_norms[i][0])
        for j in range(len(C)):
            C[j][i] = C[j][i] / scale

    ## W = intersection of sampled rows and columns
    W = []

    for i in range(r):
        temp = []
        for j in range(r):
            temp.append(input_matrix[row_Forb_norms[i][1]][col_Forb_norms[j][1]])
        W.append(temp)

    W = np.asarray(W, dtype=np.float32)
    temp = svd_helper(W)
    X, sigma, Yt = temp.decompose()

    for i in range(len(sigma)):
        for j in range(len(sigma[0])):
            if sigma[i][j] != 0:
                sigma[i][j] = 1 / sigma[i][j]

    Y = np.transpose(Yt)
    Xt = np.transpose(X)
    U = np.dot(Y, np.dot(np.dot(sigma, sigma), Xt))

    return (C, U, R)   


def main():
    """
    Main Function
    """

    file1 = open("../utility",'rb')
    orig_matrix = pickle.load(file1)


    no_rows = len(orig_matrix)
    no_cols = len(orig_matrix[0])
    print("\n For CUR\n ")
    tic= time.time()
    picking_factor = 4
    C, U, R = process_cur(orig_matrix, picking_factor)
    result = np.matmul(C, np.matmul(U,R))

    ## Removing extremes
    for i in range(len(result)):
        for j in range(len(result[0])):
            if result[i][j] > 5:
                result[i][j] = 0
            elif result[i][j] < 0:
                if ((abs(result[i][j]) > 0) and (abs(result[i][j]) < 5)):
                    result[i][j] = abs(result[i][j])
                else:
                    result[i][j] = 0     

    rmse = get_rms_error(orig_matrix, result)
    n = no_rows*no_cols
    spear_correlation = 1 - ((6* rmse*rmse*n*n)/(n* (math.pow(n,2) - 1)))
    avg_precision = precision_k(orig_matrix,result)

    print(result)
    
    toc = time.time() 
    print("RMSE = ",rmse)
    print("spear correlation = ", spear_correlation)
    print("Precision @ top k= ",avg_precision)
    print("time taken = ", toc - tic, " s")



    tic = time.time()
    print("\n For CUR with atleast 90 percent energy retained\n ")
    C_new, U_new, R_new = process_cur(orig_matrix, picking_factor )
    result_90_energy = np.matmul(C_new, np.matmul(U_new,R_new))

    ## Removing extremes
    for i in range(len(result_90_energy)):
        for j in range(len(result_90_energy[0])):
            if result_90_energy[i][j] > 5:
                result_90_energy[i][j] = 0
            elif result_90_energy[i][j] < 0:
                if ((abs(result_90_energy[i][j]) > 0) and (abs(result_90_energy[i][j]) < 5)):
                    result_90_energy[i][j] = abs(result_90_energy[i][j])
                else:
                    result_90_energy[i][j] = 0     

    print(result_90_energy)
    rmse1 = get_rms_error(orig_matrix, result_90_energy)
    n = 943*1683
    spear_correlation1 = 1 - ((6* rmse*rmse*n*n)/(n* (math.pow(n,2) - 1)))
    avg_precision1 = precision_k(orig_matrix,result)
    toc = time.time()
    
    print("RMSE = ", rmse1)
    print("spear correlation = ", spear_correlation1)
    print("Precision @ top k= ",avg_precision1)
    print("time taken = ", toc - tic, " s")





if __name__ == "__main__":
    main()


'''
Sample data input    
orig_matrix =  [[1, 1, 1, 0, 0],
                [3, 3, 3, 0, 0],
                [4, 4, 4, 0, 0],
                [5, 5, 5, 0, 0],
                [0, 0, 0, 4, 4],
                [0, 0, 0, 5, 5],
                [0, 0, 0, 2, 2]]
    
'''