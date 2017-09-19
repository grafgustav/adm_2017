import numpy as np
import time


class Recommenders(object):
    """
    How to structure the recommender systems and how to call them from the main method?
    I suggest we have static methods for each recommendation technique taking a user and a movie as input.
    The method then calculates a score for this cell and returns it.
    WARNING: If a recommendation technique requires a lot of pre-calculations, it might be better to use the
    object structure of this class to store values and reuse them for various calculations.
    """
    _global_mean = -1
    _user_predictions = np.zeros(6040)
    _movie_predictions = np.zeros(3952)

    def global_recommender_train(self, data):
        """
        Take the mean of all available ratings and use it as a prediction.
        """
        print("Global Mean calculated: {}".format(self._global_mean))
        total_sum = data[:,2].sum()
        count = np.shape(data)[0]
        self._global_mean = total_sum/count
        return total_sum/count

    def global_recommender_test(self, data, user_id, movie_id):
        return self._global_mean

    def user_recommender_train(self, data):
        # we get the data as a 750000x3 matrix
        for userId in np.arange(6040):
            self._user_predictions[userId] = self._user_recommender_train_user(data, userId+1)

    def _user_recommender_train_user(self, data, user_id):
        user_data = np.where(data[:,2] == user_id)
        if np.size(user_data) > 0:
            return np.sum(user_data, axis=0)[2]/user_data[0].shape[0]
        else:
            return 0

    def user_recommender_test(self, data, user_id, movie_id):
        return self._user_predictions[user_id-1]

    def movie_recommender_train(self, data):
        # we get the data as a 750000x3 matrix
        for movie_id in np.arange(3952):
            self._movie_predictions[movie_id] = self._user_recommender_train_user(data, movie_id+1)
        print("User Mean calculated")

    def _movie_recommender_train_user(self, data, movie_id):
        movie_data = np.where(data[:,2] == movie_id)
        if np.size(movie_data) > 0:
            return np.sum(movie_data, axis=0)[2]/movie_data[0].shape[0]
        else:
            return 0

    def movie_recommender_test(self, data, user_id, movie_id):
        return self._movie_predictions[movie_id-1]