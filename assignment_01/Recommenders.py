import numpy as np


class Recommenders(object):
    """
    How to structure the recommender systems and how to call them from the main method?
    I suggest we have static methods for each recommendation technique taking a user and a movie as input.
    The method then calculates a score for this cell and returns it.
    WARNING: If a recommendation technique requires a lot of pre-calculations, it might be better to use the
    object structure of this class to store values and reuse them for various calculations.
    """
    _global_mean = -1
    _user_predictions = np.array([])

    def global_recommender_train(self, data):
        """
        Take the mean of all available ratings and use it as a prediction.
        """
        if self._global_mean > 0:
            return self._global_mean
        else:
            total_sum = data.sum()
            count = np.count_nonzero(data)
            return total_sum/count
        print("Global Mean calculated")

    def global_recommender_test(self, data, user_id, movie_id):
        return self._global_mean

    def user_recommender_train(self, data):
        # we get the data as a 3x750000 matrix
        user_predictions = []
        for userId in np.arange(6040):
            user_predictions.append(self._user_recommender_train_user(data, userId+1))
        self._user_predictions = np.array(user_predictions)
        print("User Mean calculated")

    def _user_recommender_train_user(self, data, user_id):
        #print(data[:10])
        user_data = np.array([x for x in data if x[0] == user_id])
        #print(user_data)
        return np.sum(user_data, axis=0)[2]/user_data.shape[0]

    def user_recommender_test(self, data, user_id, movie_id):
        return self._user_predictions[user_id]