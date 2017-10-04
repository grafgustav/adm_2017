# Implement functionality to simulate the data stream here
import numpy as np


def main_function():
    # define parameters used
    length = 10e10

    print("Starting processing")
    for i in simulate_data_stream(length):
        print(i)


def simulate_data_stream(length):
    # generator yielding bits? is this slow?
    for i in np.arange(length):
        yield i

main_function()