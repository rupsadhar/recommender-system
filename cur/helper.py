import os
import cmath
import math
import pandas
import numpy


class svd_helper:

    def __init__(self, matrix, k=3):
        self.hidden_factor = k
        self.utility_matrix = matrix

    def decompose(self):
        w_1_1 = self.utility_matrix.dot(self.utility_matrix.T)
        e_value_1_1, e_vector_1_1 = numpy.linalg.eigh(w_1_1)

        w_1_2 = self.utility_matrix.T.dot(self.utility_matrix)
        e_value_1_2, e_vector_1_2 = numpy.linalg.eigh(w_1_2)

        idx_1_1 = e_value_1_1.argsort()[::-1]
        e_value_1_1 = e_value_1_1[idx_1_1]
        e_vector_1_1 = e_vector_1_1[:, idx_1_1]

        idx_1_2 = e_value_1_2.argsort()[::-1]
        e_value_1_2 = e_value_1_2[idx_1_2]
        e_vector_1_2 = e_vector_1_2[:, idx_1_2]

        self.U = e_vector_1_1
        temp = numpy.diag(numpy.array(
            [cmath.sqrt(x).real for x in e_value_1_2]))
        self.S = numpy.zeros_like(self.utility_matrix).astype(numpy.float64)
        self.S[:temp.shape[0], :temp.shape[1]] = temp
        self.V = e_vector_1_2.T

        return self.U, self.S, self.V

    def reconstruct(self):
        self.reconstructed_matrix = numpy.matmul(
            numpy.matmul(self.U, self.S), self.V)

    def get_rms_error(self):
        error = 0
        N = len(self.reconstructed_matrix)
        M = len(self.reconstructed_matrix[0])
        for i in range(len(self.reconstructed_matrix)):
            for j in range(len(self.utility_matrix[i])):
                error += math.pow(
                    self.reconstructed_matrix[i, j] - self.utility_matrix[i, j], 2)

        return math.sqrt(error/(N*M))
