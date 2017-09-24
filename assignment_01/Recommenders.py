import numpy as np
import math


class Recommenders(object):
    _global_mean = -1
    _user_predictions = np.zeros(6040)
    _movie_predictions = np.zeros(3952)
    _lin_coefficients = []
    _u_matrix = np.array([])
    _m_matrix = np.array([])
    _eeta = 0.001
    _lambda = 0.01
    _error_diff_threshold = 0.005

    def global_recommender_train(self, data):
        total_sum = data[:, 2].sum()
        count = np.shape(data)[0]
        self._global_mean = total_sum/count
        return total_sum/count

    def global_recommender_test(self, user_id, movie_id):
        return self._global_mean

    def user_recommender_train(self, data):
        for userId in np.arange(6040):
            self._user_predictions[userId] = self._user_recommender_train_user(data, userId+1)

    def _user_recommender_train_user(self, data, user_id):
        user_data = data[np.where(data[:, 0] == user_id)[0]]
        if np.size(user_data) > 0:
            return np.sum(user_data, axis=0)[2]/user_data.shape[0]
        else:
            return 0

    def user_recommender_test(self, user_id, movie_id):
        return self._user_predictions[user_id-1]

    def movie_recommender_train(self, data):
        # we get the data as a 750000x3 matrix
        for movie_id in np.arange(3952):
            self._movie_predictions[movie_id] = self._movie_recommender_train_movie(data, movie_id+1)

    def _movie_recommender_train_movie(self, data, movie_id):
        movie_data = data[np.where(data[:, 1] == movie_id)[0]]
        if np.size(movie_data) > 0:
            return np.sum(movie_data, axis=0)[2]/movie_data.shape[0]
        else:
            return 0

    def movie_recommender_test(self, user_id, movie_id):
        return self._movie_predictions[movie_id-1]

    def weighted_recommender_train(self, data):
        # movie and user recommenders should have run before
        matr = np.array([[self._user_predictions[rating[0]-1], self._movie_predictions[rating[1]-1], 1] for rating in data])
        y = data[:, 2]
        S = np.linalg.lstsq(matr, y)
        self._lin_coefficients = S[0]

    def weighted_recommender_test(self, user_id, movie_id):
        a = self.movie_recommender_test(user_id, movie_id)
        b = self.user_recommender_test(user_id, movie_id)
        alpha, beta, gamma = self._lin_coefficients
        return a*alpha + b*beta + gamma

    def matrix_factorization_train(self, data):
        self._u_matrix = np.random.uniform(low=-1, high=1, size=6040 * 100).reshape((6040, 100))
        self._m_matrix = np.random.uniform(low=-1, high=1, size=100 * 3952).reshape((100, 3952))

        rmse = 0
        last_rmse = 100
        counter = 0
        while abs(rmse - last_rmse) > self._error_diff_threshold:
            counter += 1
            rmse_sum = 0
            for rating in data:
                i = rating[0]-1
                j = rating[1]-1
                r = rating[2]
                pred = np.dot(self._u_matrix[i, :], self._m_matrix[:, j])
                error = r - pred if r > 0 else 0
                u_grad_new = self._u_matrix[i, :] + self._eeta * (2*error*self._m_matrix[:, j] - self._lambda * self._u_matrix[i, :])
                m_grad_new = self._m_matrix[:, j] + self._eeta * (2 * error * self._u_matrix[i, :] - self._lambda * self._m_matrix[:, j])
                self._u_matrix[i, :] = u_grad_new
                self._m_matrix[:, j] = m_grad_new
                rmse_sum += error * error
            last_rmse = rmse
            rmse = math.sqrt(rmse_sum / len(data))
            print("Error: " + str(rmse))
        print("Matrix Factorization trained in " + str(counter) + " iterations.")

    def matrix_factorization_test(self, user_id, movie_id):
        pred = np.dot(self._u_matrix[user_id-1, :], self._m_matrix[:, movie_id-1])
        return pred