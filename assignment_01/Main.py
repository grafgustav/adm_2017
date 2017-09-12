import time
import numpy as np


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


# read the data set and get some ratings
read_data_set()