# Implement functionality to simulate the data stream here
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import hashlib


def main_function():
    # define parameters used
    length = 1000000
    hash_table_length = 32
    max_int = math.pow(2, hash_table_length)-1

    np.random.seed(2018)

    print("Starting processing")

    main_probabilistic_counting()
    # main_log_log()
    # main_flajolet_martin()


def main_flajolet_martin():
    hash_table_length = 32
    hash_table = np.zeros(hash_table_length)
    counter = 0
    for i in get_pseudo_hash_function():
        bit_string = conv_array_to_string(i)
        k = count_trailing_zeroes(bit_string)
        hash_table[k] = 1
        counter += 1
    estimate = calculate_r(hash_table)
    print(estimate)


def conv_array_to_string(array):
    return ''.join([str(num) for num in array])


def main_log_log():
    for k in range(20):
        if k == 0:
            continue
        estimate = count_with_log_log(k)
        print("There are " + str(estimate) + " unique elements in the stream")


def simulate_data_stream(length, max_int):
    # generator yielding bits? is this slow?
    for i in np.arange(length):
        yield np.random.randint(0, max_int)


def count_trailing_zeroes(_input):
    _str = str(_input)
    return len(_str) - len(_str.rstrip('0'))


def calculate_r(hash_table):
    phi = 0.77351
    r = 0
    for i in np.arange(64):
        if hash_table[i] == 0:
            r = i
            break

    print(hash_table)

    return round(math.pow(2, r)/1, 0)


def my_hash_function(i, num_buckets):
    return int(i % num_buckets)


# a bitstring of length k will be the index for the buckets
def count_with_log_log(k):
    # number of buckets: 2^k
    num_buckets = math.pow(2, k)
    hash_table = np.zeros(int(math.pow(2, k)))
    for h in get_pseudo_hash_function():
        i = conv_array_to_string(h)
        j = i[:k]
        hash_table[int(j, 2)] = max(hash_table[int(j, 2)], get_index_of_first_one(i[k:]))

    factor = 0.39701
    if k <= 4:
        factor = factor - (2 * math.pow(math.pi, 2) + math.pow(math.log2(2), 2))/(48*num_buckets)

    return num_buckets * factor * math.pow(2, float(sum(hash_table)) / num_buckets)


def get_index_of_first_one(i):
    if len(i) < 32:
        return 32 - len(i) + 1
    for counter in range(len(i)):
        if i[counter] == '1':
            return counter+1


def get_pseudo_hash_function():
    # pseudo hash function returning a bit array of length 32
    hash_function = np.random.randint(0, 2, 8388608).reshape((32, 262144))
    for hash_value in hash_function.T:
        yield hash_value


def main_probabilistic_counting():
    phi = 0.77351
    nmap = 64
    maxlength = 32
    bitmaps = np.zeros((nmap, maxlength))
    for x in get_pseudo_hash_function():
        num = int(conv_array_to_string(x), 2)
        alpha = num % nmap
        index = get_index_of_first_one(bin(num // maxlength).split("b")[1])-1
        bitmaps[alpha, index] = 1
    S = 0
    for i in range(nmap):
        R = 0
        print(bitmaps[i, :])
        while (R <= maxlength) and (bitmaps[i, R] == 1):
            R += 1
            S += R

    Z = math.trunc(nmap/phi*math.pow(2, S / nmap))
    print(Z)


main_function()
