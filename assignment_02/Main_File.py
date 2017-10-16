import numpy as np
import math


def main_function():
    np.random.seed(2017)
    true_count = 262144 # for loglog and flajolet-martin
    alt_true_count = 4096 # for probabilistic counting

    print("Starting processing")
    print("Elements to process by LogLog and Flajolet-Martin: " + str(true_count))
    print("Elements to process by Probabilistic Counting: " + str(alt_true_count))

    print("Probabilistic counting estimate:")
    pc_estimate = main_probabilistic_counting_reduced()
    print(pc_estimate)
    print("RAE probabilistic counting: " + str(abs(pc_estimate - alt_true_count)/alt_true_count))
    print("log-log estimate:")
    ll_ks, ll_estimates = main_log_log()
    print("For k-values: " + str(ll_ks) + " estimates: " + str(ll_estimates) + "were achieved")
    rae_list_ll = [abs(ll_estimate - true_count)/true_count for ll_estimate in ll_estimates]
    print("RAE log-log estimates: " + str(rae_list_ll))
    print("Flajolet-Martin estimate:")
    fm_estimate = main_flajolet_martin()
    print(fm_estimate)
    print("RAE Flajolet-Martin: " + str(abs(fm_estimate - true_count)/true_count))
    print("Finished processing")


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
    return estimate


def conv_array_to_string(array):
    return ''.join([str(num) for num in array])


def main_log_log():
    ks = range(15)
    estimates = []
    for k in ks:
        if k == 0:
            continue
        estimates.append(count_with_log_log(k))
    return ks, estimates


def simulate_data_stream(length, max_int):
    # generator yielding bits? is this slow?
    for i in np.arange(length):
        yield np.random.randint(0, max_int)


def count_trailing_zeroes(_input):
    _str = str(_input)
    return len(_str) - len(_str.rstrip('0'))


def calculate_r(hash_table):
    r = 0
    for i in np.arange(64):
        if hash_table[i] == 0:
            r = i + 1
            break
    # print(hash_table)
    return round(math.pow(2, r), 0)


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
    # fill up with leading zeros if bitstring is too short
    if len(i) < 32:
        return 32 - len(i) + 1
    for counter in range(len(i)):
        if i[counter] == '1':
            return counter+1


def get_index_of_first_zero(i):
    # fill up with leading zeros if bitstring is too short
    for counter in range(len(i)):
        if i[len(i) - counter - 1] == '0':
            return counter + 1


def get_index_of_first_one_from_right(i):
    for counter in range(len(i)):
        if i[len(i)-1-counter] == '1':
            return counter + 1


def get_pseudo_hash_function():
    # pseudo hash function returning a bit array of length 32
    hash_function = np.random.randint(0, 2, 8388608).reshape((32, 262144))
    for hash_value in hash_function.T:
        yield hash_value


def main_probabilistic_counting_reduced():
    phi = 0.77351
    nmap = 64 # amount of bitmap vectors
    bitmap_length = 32
    bitmap = np.zeros(bitmap_length*nmap).reshape((nmap, bitmap_length)) # nmap bitmap vectors, shape (64,32)
    counter = 0
    for x in get_pseudo_hash_function():
        _map = counter % 64 # _map in [0, 63]
        index = get_index_of_first_one(conv_array_to_string(x)) - 1
        bitmap[_map, index] = 1
        counter += 1

    p_sum = 0
    for p_index in range(nmap):
        p_sum += np.where(bitmap[p_index, :] == 0)[0][0]

    p_avg = p_sum / nmap
    estimate = (1/phi)*math.pow(2, p_avg)
    return estimate


main_function()
