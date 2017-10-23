import numpy as np
import scipy.sparse as sps
import _pickle as pickle
import sys
import pandas as pd
from scipy.spatial.distance import pdist, jaccard
from scipy.spatial.distance import squareform
from collections import defaultdict
import random
import time


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
    n = 50

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
            start_time = time.time()
            permutation = get_permutation(nr_movies)
            print(time.time() - start_time)
            start_time = time.time()
            signature_matrix[i] = build_signature(data_matrix, permutation)
            print(signature_matrix[i])
            print(time.time() - start_time)
            print(i)
        pickle.dump(signature_matrix, open("sig_matrix2.p", "wb"))
    else:
        signature_matrix = pickle.load(open("sig_matrix2.p", "rb"))
    print(signature_matrix.shape)
    candidate_pairs = execute_lsh(signature_matrix)
    results = test_jaccard_similarity(candidate_pairs, data_matrix)
    print(len(results))
    print_results(results)
    print("Finished processing!")


def load_data():
    print("Loading data")
    data = np.load("user_movie.npy")
    print("Data loaded")
    return data


def build_sparse_matrix(data):
    print("Building sparse matrix")
    # cols = movies, rows = users
    cols, rows = 17770, 103703
    np_matr = np.zeros(rows*cols).reshape((rows, cols))
    # adjust index
    np_matr[data[:, 0]-1, data[:, 1]-1] = 1
    print("Sparse matrix built")
    return sps.csr_matrix(np_matr)


def get_permutation(max_index):
    # print("Building permutation")
    # n = number of hash functions, max_index = length of vector
    return np.random.permutation(max_index)


def build_signature(data_matrix, permutation):
    # data_matrix: 17770x103703, permutation: len(17770)
    # build the signature over one permutation for every user
    # print("Building signature")
    n = data_matrix.shape[0]
    signature = np.zeros(n)
    permuted_matrix = data_matrix[:, permutation] # takes about 1.6s

    # TODO: this stuff can surely be folded
    for i in range(n):
        signature[i] = np.min(permuted_matrix[i].indices)
    # print("Signature built")
    return signature


def execute_lsh(signature_matrix):
    # Locality Sensitive Hashing
    print("Executing the LSH function")
    nr_buckets, band_size = 100, 10
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
            band_hash = tuple(band)
            # 3. check if hash bucket is used
            if hash_dic.get(band_hash):
                # 4. match users together who map bands to the same bucket in candidate column pair list
                for can in hash_dic.get(band_hash):
                    # TODO: Eliminate duplicates
                    if can < user_index:
                        candidate_list.append((can, user_index))
                    else:
                        candidate_list.append((user_index, can))

                hash_dic[band_hash].append(user_index)
            else:
                # check for duplicates
                hash_dic[band_hash] = [user_index]

            user_index += 1

    # candidate list is filled with candidates that have at least one band sufficiently similar
    # next step: calculate exact jaccard similarity (using signatures, or using the full data set?)

    print(len(candidate_list))
    print("Done")
    return candidate_list


def test_jaccard_similarity(candidate_pairs, data_matrix):
    print("Test Candidate pairs for actual Jaccard similarity")
    results = []
    for can1, can2 in candidate_pairs:
        jaccard_similarity = sum((data_matrix[can1, :].toarray().astype(int) & data_matrix[can2, :].toarray().astype(int))[0]) /\
                             sum((data_matrix[can1, :].toarray().astype(int) | data_matrix[can2, :].toarray().astype(int))[0])
        if jaccard_similarity >= 0.5:
            if (can1, can2) not in results:
                results.append((can1, can2))
    print(results)
    results.sort()
    return results


def execute_lsh_2(S):
    print(S.shape)
    S = S[0:500, :]

    n = 50
    B = 10
    R = n / B

    a = np.repeat(list(range(B)), R)
    a = np.array(random.sample(list(a), len(a)))
    print(a)

    Test = pd.DataFrame(S[np.where(a == 1)[0]])
    # v = Test.values
    # lt = pd.DataFrame((v[:, None] == v.T))
    # np.fill_diagonal(lt.values, 0)
    res = 1 - pdist(Test.T, 'jaccard')
    squareform(res)
    distance = np.triu(pd.DataFrame(squareform(res), index=Test.T.index, columns=Test.T.index), k=0)

    d = defaultdict(list)
    for pos, val in np.ndenumerate(distance):
        if val:
            d[val].append((pos[0], pos[1]))

    return d


def check_if_pair_is_similar(sig1, sig2):
    return sig1 == sig2


def output_tuple_to_txt(tuple):
    # tuple consists of (lower id, higher id)
    with open("results.txt", "a") as myfile:
        myfile.write(str(tuple))


def print_results(results):
    for tpl in results:
        output_tuple_to_txt(tpl)


# if set_options():
main_function()
# else:
#   print("Argument list invalid")
#   print("Usage: Main.py [random seed] ["path-to-user-movie.npy"])
