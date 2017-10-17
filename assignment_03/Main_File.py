import numpy as np
import scipy.sparse as sps


def main_function():
    n = 10

    print("Starting processing!")
    data = load_data()
    print(data.shape)
    data_matr = build_sparse_matrix(data)
    rows = len(data_matr)
    print(rows)
    for i in range(n):
        permutation = get_permutation(n, rows)
        signature = build_signature(data_matr, permutation)

    print("Finished processing!")


def load_data():
    print("Loading data")
    data = np.load("user_movie.npy")
    print("Data loaded")
    return data


def build_sparse_matrix(data):
    print("Building sparse matrix")
    rows, cols = 17770, 103703
    # sps_acc = sps.coo_matrix((rows, cols))
    # sps_acc[data[0]-1, data[1]-1] = 1
    np_matr = np.zeros(rows*cols).reshape((rows, cols))
    np_matr[data[:, 1]-1, data[:, 0]-1] = 1
    print("Sparse matrix built")
    return np_matr


def get_permutation(n, max_index):
    print("Building permutation matrix")
    # n = number of hash functions, max_index = length of vector
    return np.random.permutation(max_index)


def build_signature(data_matrix, permutation):
    # data_matrix: 103703x17770, permutation: len(17770)
    print("Building signature matrix")
    signature = np.zeros(len(permutation))

    print("Signature matrix built")
    return signature

main_function()