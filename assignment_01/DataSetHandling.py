import numpy as np
import time
from sklearn.model_selection import KFold


class DataSetHandling(object):
    """
    This class provides all necessary functions for accessing the data set provided by the ml-1m team.
    The main set of interest is the ratings.dat data, which is why this class only provides methods to
    access these. Loading the data from the other sets to develop more sophisticated
    recommender systems works analogously.

    To use this class, create an instance of it by (implicitly) calling the constructor. The data set
    will be loaded into memory as a matrix and stored inside this object. Access the matrix by using
    the adequate method.
    """

    # init data as empty matrix
    _data = np.matrix([])
    _kfold_iter = 0
    _kfold_obj = KFold(n_splits=5, shuffle=True)

    def __init__(self, seed=1):
        # read data set on construction of object
        self._read_data_set()
        super.__init__()
        return

    def _read_data_set(self):
        """
        Reads the ratings data set and executes the cross-validation shuffling.
        """
        start_time = time.time()
        # shape: 6040 users, 3952 movies

        ratings = np.genfromtxt("ratings.dat", usecols=(0, 1, 2), delimiter='::', dtype='int')
        print(ratings.shape)

        self._data = ratings

        print("Loading the dataset took: " + str(time.time() - start_time) + " seconds")

        return

    def get_data_set(self):
        """
        Returns the whole loaded data set.
        :return: Matrix in the size of the ratings data set.
        """
        return self._data

    def get_next_kfold(self):
        """
        Returns the next tuple of indices for training and test set for cross-validation.

        Example:
            iter_tuple = dh.get_next_kfold()
            while  iter_tuple != -1:
                train(dh.get_data_set[iter_tuple[0]])
                test(dh.get_data_set[iter_tuple[1]])
                iter_tuple = dh.get_next_kfold()
        :return: (train_indices, test_indices)
        """
        if self._kfold_iter < 5:
            result_tuple = self._kfold_obj.split(self._data)[self._kfold_iter]
            self._kfold_iter += 1
            return result_tuple
        else:
            # some kind of error?
            return -1

