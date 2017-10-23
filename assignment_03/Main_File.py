import numpy as np
import scipy.sparse as sps
import _pickle as pickle
import sys

file_path = ""


def set_options():
    arguments = sys.argv
    if len(arguments) == 2:
        np.random.seed(arguments[0])
        np.set_printoptions(edgeitems=10)
        file_path = arguments[1]
        return True
    else:
        return False


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
    execute_lsh(signature_matrix)
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


def execute_lsh(signature_matrix):
    # Locality Sensitive Hashing
    print("Executing the LSH function")
    nr_buckets, band_size = 100, 50
    nr_signatures = signature_matrix.shape[0]
    nr_bands = int(nr_signatures / band_size)

    hash_dic = dict()
    candidate_list = []
    # 1. divide user signature into bands
    for i in range(nr_bands):
        user_index = 0
        for user in signature_matrix.T:
            # user is min hash vector
            band = user[i*band_size:(i+1)*band_size]
            # 2. apply hash function to band
            band_hash = my_hash(band)
            # 3. check if hash bucket is used
            if hash_dic.get(band_hash):
                # 4. match users together who map bands to the same bucket in candidate column pair list
                candidate_list.append((hash_dic.get(band_hash), user_index))
            else:
                hash_dic[band_hash] = user_index

            user_index += 1

    print(candidate_list)
    print(len(candidate_list))
    candidate_filter = [check_if_pair_is_similar(signature_matrix[:, sig1], signature_matrix[:, sig2])
                        for sig1, sig2 in candidate_list]
    print(candidate_filter)
    #print(len(candidate_list[candidate_filter]))
    return candidate_list[candidate_filter]


def check_if_pair_is_similar(sig1, sig2):
    return sig1 == sig2


def my_hash(l):
    # 104729 prime #10,000
    # bad hash function, because sum can be very ambiguous
    return sum(l) % 104729


def output_tuple_to_txt(tuple):
    # tuple consists of (lower id, higher id)
    with open("results.txt", "a") as myfile:
        myfile.write(str(tuple))


# if set_options():
main_function()
# else:
#   print("Argument list invalid")
#   print("Usage: Main.py [random seed] ["path-to-user-movie.npy"])
