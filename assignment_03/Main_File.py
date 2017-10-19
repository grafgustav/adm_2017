import numpy as np
import scipy.sparse as sps
import _pickle as pickle


def main_function():
    # to persist the data for quick access, set to persist=True, to load it in later executions, set persist=False
    persist_data = False
    persist_signature_matrix = False

    # length of signatures
    n = 150

    print("Starting processing!")
    if persist_data:
        data = load_data()
        print(data.shape)
        data_matrix = build_sparse_matrix(data)
        pickle.dump(data_matrix, open("data.p", "wb"))
    else:
        data_matrix = pickle.load(open("data.p", "rb"))

    nr_movies = data_matrix.shape[1]
    nr_users = data_matrix.shape[0]
    print((nr_movies, nr_users))
    if persist_signature_matrix:
        signature_matrix = np.zeros((n, nr_users))
        for i in range(n):
            permutation = get_permutation(nr_movies)
            signature_matrix[i] = build_signature(data_matrix, permutation)
            print(i)
        pickle.dump(signature_matrix, open("sig_matrix.p", "wb"))
    else:
        signature_matrix = pickle.load(open("sig_matrix.p", "rb"))
    print(signature_matrix.shape)
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
    return sps.csr_matrix(np_matr)


def get_permutation(max_index):
    # print("Building permutation")
    # n = number of hash functions, max_index = length of vector
    return np.random.permutation(max_index)


def build_signature(data_matrix, permutation):
    # data_matrix: 103703x17770, permutation: len(17770)
    # build the signature over one permutation for every user
    # print("Building signature")
    n = data_matrix.shape[0]
    signature = np.zeros(n)
    for i in range(n):
        signature[i] = np.min(permutation[data_matrix[i, :].nonzero()[1]])
    # print("Signature built")
    return signature


np.random.seed(2017)
np.set_printoptions(edgeitems=10)
main_function()
