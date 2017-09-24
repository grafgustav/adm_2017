import time
import math
import numpy as np
from adm_2017.assignment_01.DataSetHandling import DataSetHandling
from adm_2017.assignment_01.Recommenders import Recommenders


def my_main():
    # run what you need
    dh = DataSetHandling()
    recommender = Recommenders()
    kf = dh.get_kfold_obj()

    g_times = []
    u_times = []
    m_times = []
    l_times = []
    matrix_times = []

    g_rmse_results = []
    u_rmse_results = []
    m_rmse_results = []
    l_rmse_results = []
    matrix_rmse_results = []

    g_mae_results = []
    u_mae_results = []
    m_mae_results = []
    l_mae_results = []
    matrix_mae_results = []

    np.random.seed(42)

    for train, test in kf:
        # calculate and set the global mean of the object
        print("A Training the global recommender")
        tmp_time = time.time()
        recommender.global_recommender_train(dh.get_data_set()[train])
        g_times.append(time.time() - tmp_time)
        print("B Training the user recommender")
        tmp_time = time.time()
        recommender.user_recommender_train(dh.get_data_set()[train])
        u_times.append(time.time() - tmp_time)
        print("C Training the movie recommender")
        tmp_time = time.time()
        recommender.movie_recommender_train(dh.get_data_set()[train])
        m_times.append(time.time() - tmp_time)
        print("D Training the Linear Regression Model")
        tmp_time = time.time()
        recommender.weighted_recommender_train(dh.get_data_set()[train])
        l_times.append(time.time() - tmp_time)
        print("E Training the Matrix Factorization")
        tmp_time = time.time()
        recommender.matrix_factorization_train(dh.get_data_set()[train])
        matrix_times.append(time.time() - tmp_time)

        gmse_count = 0
        g_gmse_sum = 0
        u_gmse_sum = 0
        m_gmse_sum = 0
        l_gmse_sum = 0
        matrix_gmse_sum = 0

        g_mae_sum = 0
        u_mae_sum = 0
        m_mae_sum = 0
        l_mae_sum = 0
        matrix_mae_sum = 0

        print("Starting the tests")
        for rating in dh.get_data_set()[test]:
            act_val = rating[2]
            g_pred = recommender.global_recommender_test(rating[0], rating[1])
            u_pred = recommender.user_recommender_test(rating[0], rating[1])
            m_pred = recommender.movie_recommender_test(rating[0], rating[1])
            l_pred = recommender.weighted_recommender_test(rating[0], rating[1])
            matrix_pred = recommender.matrix_factorization_test(rating[0], rating[1])

            g_mse = pow(act_val - g_pred, 2)
            u_mse = pow(act_val - u_pred, 2)
            m_mse = pow(act_val - m_pred, 2)
            l_mse = pow(act_val - l_pred, 2)
            matrix_mse = pow(act_val - matrix_pred, 2)

            g_gmse_sum += g_mse
            u_gmse_sum += u_mse
            m_gmse_sum += m_mse
            l_gmse_sum += l_mse
            matrix_gmse_sum += matrix_mse

            g_mae_sum += abs(act_val - g_pred)
            u_mae_sum += abs(act_val - u_pred)
            m_mae_sum += abs(act_val - m_pred)
            l_mae_sum += abs(act_val - l_pred)
            matrix_mae_sum += abs(act_val - matrix_pred)

            gmse_count += 1

        g_rmse_results.append(math.sqrt(g_gmse_sum / gmse_count))
        u_rmse_results.append(math.sqrt(u_gmse_sum / gmse_count))
        m_rmse_results.append(math.sqrt(m_gmse_sum / gmse_count))
        l_rmse_results.append(math.sqrt(l_gmse_sum / gmse_count))
        matrix_rmse_results.append(math.sqrt(matrix_gmse_sum / gmse_count))

        g_mae_results.append(g_mae_sum / gmse_count)
        u_mae_results.append(u_mae_sum / gmse_count)
        m_mae_results.append(g_mae_sum / gmse_count)
        l_mae_results.append(l_mae_sum / gmse_count)
        matrix_mae_results.append(matrix_mae_sum / gmse_count)

    print("Global Mean Recommender Results:")
    print("RMSE: " + str(g_rmse_results))
    print("MAE: " + str(g_mae_results))
    print("Times: " + str(g_times))

    print("User Mean Recommender Results:")
    print("RMSE: " + str(u_rmse_results))
    print("MAE: " + str(u_mae_results))
    print("Times: " + str(u_times))

    print("Movie Mean Recommender Results:")
    print("RMSE: " + str(m_rmse_results))
    print("MAE: " + str(m_mae_results))
    print("Times: " + str(m_times))

    print("Linear Regression Model Recommender Results:")
    print("RMSE: " + str(l_rmse_results))
    print("MAE: " + str(l_mae_results))
    print("Times: " + str(l_times))

    print("Matrix Factorization Results:")
    print("RMSE: " + str(matrix_rmse_results))
    print("MAE: " + str(matrix_mae_results))
    print("Times: " + str(matrix_times))

my_main()