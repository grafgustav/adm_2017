import time
import math


def main():
    # run what you need
    test_test_set()


def test_test_set():
    # build model from training set and test it on test set?
    print("Testing")


def calculate_rmse(estimate):
    """
    Calculate the Root Mean Squared Error.
    :param estimate: A 5-tuple containing the errors of every cross-validation test run?
    :return:
    """
    # todo: first square the values then calculate the diff
    sum = sum(estimate)
    count = len(estimate)
    return math.sqrt(sum/count)


def calculate_mae(estimate):
    """
    Calculate the Mean Absolute Error.
    :param estimate: A 5-tuple containing the errors of every cross-validation test run.
    :return:
    """
    return abs(sum(estimate) / len(estimate))


def test_prediction_vs_test_set(test_set ,pred, user_id, movie_id):
    """
        Uses a prediction value and tests it against the existing value in the matrix
    :param pred:
    :param user_id:
    :param movie_id:
    :return: Difference between prediction and actual value
    """
    dif = abs(test_set[user_id][movie_id] - pred)
    return dif


def time_method_call(func):
    """
    Returns the used time of a function call func.
    :param func: The call to the recommender system.
    :return:
    """
    start_time = time.time()
    func()
    return time.time() - start_time