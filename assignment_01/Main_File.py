import time
import math
from adm_2017.assignment_01.DataSetHandling import DataSetHandling
from adm_2017.assignment_01.Recommenders import Recommenders


def my_main():
    # run what you need
    dh = DataSetHandling()
    recommender = Recommenders()
    kf = dh.get_kfold_obj()
    g_mse_results = []
    u_mse_results = []
    for train, test in kf:
        # calculate and set the global mean of the object
        recommender.global_recommender_train(dh.get_data_set()[train])
        recommender.user_recommender_train(dh.get_data_set()[train])
        g_gmse_sum = 0
        g_gmse_count = 0
        u_gmse_sum = 0
        u_gmse_count = 0
        for rating in dh.get_data_set()[test]:
            g_pred = recommender.global_recommender_test(dh.get_data_set(), rating[0], rating[1])
            u_pred = recommender.user_recommender_test(dh.get_data_set(), rating[0], rating[1])
            act_val = rating[2]
            g_mse = pow(act_val - g_pred, 2)
            u_mse = pow(act_val - u_pred, 2)
            g_gmse_sum += g_mse
            g_gmse_count += 1
            u_gmse_sum += u_mse
            u_gmse_count += 1


        g_gmse_sum = math.sqrt(g_gmse_sum / g_gmse_count)
        g_mse_results.append(g_gmse_sum)
        u_gmse_sum = math.sqrt(u_gmse_sum / u_gmse_count)
        u_mse_results.append(u_gmse_sum)
    print("Global Mean Recommender Results:")
    print(g_mse_results)
    print("User Mean Recommender Results:")
    print(u_mse_results)



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