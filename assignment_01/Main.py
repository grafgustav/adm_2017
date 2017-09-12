import time
import numpy as np
from sklearn.model_selection import KFold


def read_data_set():
    """
    Reads the ratings dataset and
    """
    start_time = time.time()
    ratings = np.zeros((6041, 3953))

    # load the ratings dataset
    ratings_file = open('ml-1m/ratings.dat', 'r', encoding='latin-1')
    for line in ratings_file:
        line_tuple = convert_file_entry_to_tuple(line)
        ratings[line_tuple[0]][line_tuple[1]] = line_tuple[2]
    ratings_file.close()

    print("Loading the dataset took: " + str(time.time() - start_time) + " seconds")

    np.random.seed(1)
    kf = KFold(n_splits=5, shuffle=True)
    kf.get_n_splits(ratings)
    print(kf)

    for train_index, test_index in kf.split(ratings):
        train_set = ratings[train_index]
        test_set = ratings[test_index]
        print(train_set.shape)
        print(test_set.shape)
        global_recommender(train_set, 0,0)


# takes a FileEntry as a single string as input and transforms it into a tuple of ints
def convert_file_entry_to_tuple(entry):
    string_list = entry.split("::")
    return tuple(map(parse_string_to_int, string_list))


def parse_string_to_int(string):
    return_value = 0
    try:
        return_value = int(string)
    except (ValueError, UnicodeDecodeError):
        return_value = string
    finally:
        return return_value


def global_recommender(train_set, user_id, movie_id):
    """
        Take the mean of all available ratings and use it as a prediction.
    """
    total_sum = train_set.sum()
    count = np.count_nonzero(train_set)
    print(total_sum/count)
    return total_sum/count


# read the data set and get some ratings
read_data_set()