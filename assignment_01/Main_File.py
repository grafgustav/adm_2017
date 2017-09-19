import time
import math
from adm_2017.assignment_01.DataSetHandling import DataSetHandling
from adm_2017.assignment_01.Recommenders import Recommenders


def my_main():
    # run what you need
    dh = DataSetHandling()
    recommender = Recommenders()
    kf = dh.get_kfold_obj()
    mse_results = []
    for train, test in kf:
        # calculate and set the global mean of the object
        recommender.global_recommender_train(dh.get_data_set()[train])
        gmse_sum = 0
        gmse_count = 0
        for rating in dh.get_data_set()[test]:
            pred = recommender.global_recommender_test(dh.get_data_set(), rating[0], rating[1])
            act_val = rating[2]
            mse = pow(act_val - pred, 2)
            gmse_sum += mse
            gmse_count += 1
        gmse_sum = math.sqrt(gmse_sum / gmse_count)
        mse_results.append(gmse_sum)
    print(mse_results)


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

my_main()