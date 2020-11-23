import numpy
import pandas
import pickle
import os 
import time
import math

from scipy.linalg import svd
from numpy.linalg import matrix_rank
from numpy import diag
from numpy import zeros

dataset_dir="data"
binary_dir="binaries"
package_dir="svd"

dataset=os.path.join(os.path.abspath('./'),dataset_dir)

rating_dataset=os.path.join(dataset,"ratings.dat")
binary=os.path.join(package_dir,binary_dir)
utility_matrix_bin_path=os.path.join(binary,"utility_matrix.pickle")



def load(filepath,column):
    """
        load(filepath,column) will load the dataset corresponding to the filepath as a Dataframe, with columns separated by a tab
    """
    with open(filepath,'r',encoding='ISO-8859-1') as f:
        text=str(f.read()).strip().split('\n')
        return pandas.DataFrame.from_records(
            [sentence.split('\t') for sentence in text],columns=column
        )
        
        
def assign_missing_values(input_matrix):
    """
        assign_missing_values(input_matrix) will return a matrix with missing values replaced by global avg plus the bias with respect to row averages and column averages
    """    
    matrix=numpy.asarray(input_matrix,dtype=numpy.float32)
    mean=matrix.mean()
    
    row_count,col_count=[],[]
    
    for x in range(len(input_matrix)):
        row_count.append(numpy.count_nonzero(matrix[x,:]))
    for x in range(len(matrix[0])):
        col_count.append(numpy.count_nonzero(matrix[:,x]))
        
    row_means,col_means = [],[]
    
    for x in range(len(matrix)):
        row_means.append(
            (numpy.sum(matrix[x,:])-(mean*row_count[x]))/(row_count[x]*row_count[x])
        )
    for x in range(len(matrix[0])):
        col_means.append(
            (numpy.sum(matrix[:,x])-(mean*col_count[x]))/(col_count[x]*col_count[x])
        )
    #Replace NA values
    for x in range(len(matrix)):
        for y in range(len(matrix[0])):
            if matrix[x][y]==0:
                matrix[x][y]= mean + row_means[x] + col_means[y]
                
            if matrix[x][y]>5:
                matrix[x][y]=5
                
            if matrix[x][y]<1:
                matrix[x][y]=1
    return matrix
        
def preprocess():
    """
    loads the user vs movie ratings as a matrix, and assigns values to the missing ratings

    Returns:
        [matrix]: [matrix with assigned values to the missing ratings]
    """
    dataset=load(rating_dataset,column=['uid','mid','rating','time'])
    dataset.drop(labels=["time"],axis=1,inplace=True)
    dataset=dataset.astype(int)
    
    num_users=list(dataset['uid'].unique())
    num_users.sort()
    
    num_movies=list(dataset['mid'].unique())
    num_movies.sort()
    
    utility_matrix=numpy.full((len(num_users),len(num_movies)),0)
    
    for iter in dataset.index:
        user_index=num_users.index(dataset['uid'][iter])
        movie_index=num_movies.index(dataset['mid'][iter])
        utility_matrix[user_index][movie_index]=dataset['rating'][iter]
        
    #print(utility_matrix)  
    return assign_missing_values(utility_matrix)


def calculate_svd(input_matrix):
    """
    calculates the svd from the matrix, by calculating the corresponding eigen values and eigen vectors

    Args:
        input_matrix ([matrix]): [The matrix to be decomposed to U,Sigma,V Transpose]

    Returns:
        [U,S,Vt]: [three matrices after singular value decomposition]
    """
    input_matrix=numpy.asarray(input_matrix,dtype=numpy.float32)
    
    u,s,vt=svd(input_matrix)
    
    idx_1_1=s.argsort()[::-1]
    s=s[idx_1_1]
    u=u[:,idx_1_1]
    
    idx_1_1=s.argsort()[::-1]
    s=s[idx_1_1]
    vt=vt[:,idx_1_1]
    
    return u,numpy.diag(s),vt

def calculate_svd_90(input_matrix):
    """[preserves the values in sigma matrix which sum up to 90% of the variation, and then calculates the U,Sigma,Vt by applying svd decomposition]

    Args:
        input_matrix ([matrix]): [matrix that needs to be decomposed to suv with 90% energy]

    Returns:
        [U,Sigma,Vt]: [Three matrices after suv with 90% energy]
    """
    
    input_matrix = numpy.asarray(input_matrix, dtype=numpy.float32)

    U, s, Vt = numpy.linalg.svd(input_matrix,full_matrices=False)

    sigma = numpy.zeros((input_matrix.shape[0],  input_matrix.shape[1]))
    sigma[:input_matrix.shape[1], :input_matrix.shape[1]] = numpy.diag(s)

    total = 0
    for x in range(min(len(sigma), len(sigma[0]))):
        total = total + (sigma[x][x] * sigma[x][x])

    temp = 0
    temp_total = 0
    for x in range(min(len(sigma), len(sigma[0]))):
        temp_total = temp_total + (sigma[x][x] * sigma[x][x])
        temp = temp + 1
        if (temp_total / total) > 0.9:
            break

    new_U = U[:temp, :temp]
    new_sigma = sigma[:temp, :temp]
    new_Vt = Vt[:temp, :temp]

    return new_U,new_sigma,new_Vt

def precision_k(Actual, Predicted):
    precision_list = []
    threshold = 3.5
    k = 3
    for i in range(Actual.shape[0]):
        rating_dict = {}
        for j in range(Actual.shape[1]):
            rating_dict[j] = [Predicted[i][j], Actual[i][j]]
        #print(rating_dict)
        var = {k: v for k, v in sorted(rating_dict.items(), key=lambda item: item[1], reverse=True)}
        count = 0;
        rel_recom = 0
        for i in var.keys():
            if count<k:
                count += 1
                if var[i][1] > threshold:
                    rel_recom += 1

        temp = rel_recom/k
        #print(temp)
        precision_list.append(temp)

    avg_precision = numpy.average(precision_list)
    print(avg_precision)

    return avg_precision

def predict(actual):
    """[gets predicted matrix ]

    Args:
        actual ([matrix]): [test matrix]

    Returns:
        [matrix]: [predicted matrix after svd]
    """
    U,s,Vh = numpy.linalg.svd(actual, full_matrices=False)
    assert numpy.allclose(actual, numpy.dot(U, numpy.dot(numpy.diag(s), Vh)))
    s[1:] = 0
    new_a = numpy.dot(U, numpy.dot(numpy.diag(s), Vh))
    return new_a

def predict_90(actual):
    """[gets predicted matrix for test]

    Args:
        actual ([test_matrix]): [description]

    Returns:
        [matrix]: [predicted matrix with 90 % energy]
    """
    U,s,Vh = numpy.linalg.svd(actual, full_matrices=False)
    assert numpy.allclose(actual, numpy.dot(U, numpy.dot(numpy.diag(s), Vh)))
    k=matrix_rank(s)
    k=int(k/10)
    s[k:] = 0
    new_a = numpy.dot(U, numpy.dot(numpy.diag(s), Vh))
    return new_a
        
class SVD:
    """[class to calculate and implement various functions and accuracy measures]
    """
    def __init__(self,matrix,k=3):
        """[initialised variables for the class]

        Args:
            matrix ([matrix]): [utility matrix to be passed as an object]
            k (int, optional): [default value for finding top k precision]. Defaults to 3.
        """
        self.hidden_factor=k
        self.utility_matrix=matrix
        
    def decompose(self):
        """[function to decompose the object to corresponding svd components]
        """
        A=self.utility_matrix
        u,s,vt=svd(A)
        self.U=u
        self.S=s
        self.V=vt
        
        
    def reconstruct(self):
        """[function to reconstruct the multiplied U,s,Vt after transforming to the correct sizes]
        """
        a=self.utility_matrix
        # create m x n Sigma matrix
        U, s, Vh = numpy.linalg.svd(a, full_matrices=False)
        assert numpy.allclose(a, numpy.dot(U, numpy.dot(numpy.diag(s), Vh)))
        
        s[1:] = 0
        new_a = numpy.dot(U, numpy.dot(numpy.diag(s), Vh))
        self.reconstructed_matrix=new_a
    
    def reconstruct_90(self):
        """[function to reconstruct after suv with 90% energy]
        """
        a=self.utility_matrix
        U,s,Vh = numpy.linalg.svd(a, full_matrices=False)
        assert numpy.allclose(a, numpy.dot(U, numpy.dot(numpy.diag(s), Vh)))
        k=matrix_rank(s)
        k=int(k/10)
        s[k:] = 0
        new_a = numpy.dot(U, numpy.dot(numpy.diag(s), Vh))
        self.reconstructed_matrix=new_a
    
        
    def get_rms_error(self):
        """[function to calculate rmse ]

        Returns:
            [float]: [error of original and reconstructed matrix]
        """
        error=0
        N=len(self.reconstructed_matrix)
        M=len(self.reconstructed_matrix[0])
        for i in range(len(self.reconstructed_matrix)):
            for j in range(len(self.utility_matrix[i])):
                error += math.pow(
                    self.reconstructed_matrix[i,j]-self.utility_matrix[i,j],2
                )
        return math.sqrt(error/(N*M))
    
    def get_mean_abs_error(self):
        """Returns the Mean Absolute Error of the model"""
        error = 0
        N=len(self.reconstructed_matrix)
        M=len(self.reconstructed_matrix[0])
        for i in range(len(self.reconstructed_matrix)):
            for j in range(len(self.utility_matrix[i])):
                error += abs(
                    self.reconstructed_matrix[i,j]-self.utility_matrix[i,j]
                )
        return error/(N*M)
    
    def get_size_of_matrix(self):
        """[gets the size of a matrix object]

        Returns:
            [integer]: [no of items in the matrix object]
        """
        N=len(self.reconstructed_matrix)
        M=len(self.reconstructed_matrix[0])
        return N*M
    
    def cal_spearmann_rank_correlation(self,d,n):
        """[calculates the spearmann_rank_correlation of the matrix]

        Args:
            d ([float]): [error sum of squares]
            n ([integer]): [size of the matrix]

        Returns:
            [float]: [spearmann rank correlation coefficient]
        """
        diff= 6*d*d*n/(n*n-1)
        return 1-diff


if __name__ == "__main__":
    
    file=open("utility",'rb')
    actual=pickle.load(file)
    #print(actual)
    predict_svd=predict(actual)
    #print(predicted)
    start_time=time.time()
    precision_svd=precision_k(actual, predict_svd)
    print("precision top k with svd: ",precision_svd)
    print("--- %s seconds ---" %(time.time()-start_time))
    predict_svd_90=predict_90(actual)
    start_time=time.time()
    precision_svd_90=precision_k(actual, predict_svd_90)
    print("precison top k for svd with 90% energy: ",precision_svd_90)
    print("--- %s seconds ---" %(time.time()-start_time))

    x=preprocess()
    
    a = SVD(x)
    a.decompose()
    a.reconstruct()
    print("the rmse error for svd")
    start_time=time.time()
    d1=a.get_rms_error()
    print(d1)
    print("--- %s seconds ---" %(time.time()-start_time))
    n1=a.get_size_of_matrix()
    
    print("spearmann rank correlation for svd")
    start_time=time.time()
    s1=a.cal_spearmann_rank_correlation(d1,n1)
    print(s1)
    print("--- %s seconds ---" %(time.time()-start_time))
    
    b=SVD(x)
    b.decompose()
    b.reconstruct_90()
    print("the rmse error for svd with 90 percent energy")
    start_time=time.time()
    d2=b.get_rms_error()
    print(d2)
    print("--- %s seconds ---" %(time.time()-start_time))
    
    n2=b.get_size_of_matrix()
    print("spearmann rank correlation for svd with 90% energy")
    start_time=time.time()
    s2=b.cal_spearmann_rank_correlation(d2,n2)
    print(s2)
    print("--- %s seconds ---" %(time.time()-start_time))
    
    
    
