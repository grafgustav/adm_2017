import numpy as np


class Recommenders(object):
    """
    How to structure the recommender systems and how to call them from the main method?
    I suggest we have static methods for each recommendation technique taking a user and a movie as input.
    The method then calculates a score for this cell and returns it.
    WARNING: If a recommendation technique requires a lot of pre-calculations, it might be better to use the
    object structure of this class to store values and reuse them for various calculations.
    """

    @staticmethod
    def global_recommender(data, user_id, movie_id):
        """
        Take the mean of all available ratings and use it as a prediction.
        """
        total_sum = data.sum()
        count = np.count_nonzero(data)
        return total_sum/count
