# Implement functionality to simulate the data stream here
import numpy as np
import math


def main_function():
    # define parameters used
    length = 10000
    hash_table_length = 32
    max_int = math.pow(2, hash_table_length)-1

    np.random.seed(2018)

    hash_table = np.zeros(hash_table_length)

    print("Starting processing")
    for i in simulate_data_stream(length, max_int):
        bit_string = bin(i)
        k = count_trailing_zeroes(bit_string)
        hash_table[k] = 1

    estimate = calculate_r(hash_table)
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

    return round(math.pow(2, r-1)/1, 0)


main_function()
