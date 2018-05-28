# encoding: utf8
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.vector_ar import var_model as sm
import numpy as np
from math import sqrt
import cPickle as cp

def irf(result, innovation_idx, larger_restrict_matrix, less_restrict_matrix, impulse_count, draw_count,
        restrict_period, verbose=False, plot=True):
    def get_sphere_rand():
        while True:
            trand = np.random.rand(len(sigma)) - 0.5
            return trand / sqrt(sum(trand ** 2))

    larger_restrict_array = larger_restrict_matrix[:, innovation_idx]
    less_restrict_array = less_restrict_matrix[:, innovation_idx]
    p, k = result.coefs.shape[0], result.coefs.shape[1]
    A_1 = result.coefs[0]
    sigma = result.sigma_u

    P = np.linalg.cholesky(sigma)
    B = np.linalg.inv(P)

    impulse_matrix = np.zeros((draw_count, A_1.shape[0], impulse_count))
    a_matrix = np.zeros((A_1.shape[0], draw_count))

    a = None
    for j in xrange(draw_count):
        if verbose:
            print j
        i = 0
        a = get_sphere_rand()
        while True:
            if i == impulse_count:
                break
            t_impulse = (A_1 ** (i + 1)).dot(B).dot(a)
            if i < restrict_period and (t_impulse[larger_restrict_array] > 0.0).all() and (
                    t_impulse[less_restrict_array] < 0.0).all():
                impulse_matrix[j, :, i] = t_impulse
                a_matrix[:, j] = a
                i += 1
            elif i >= restrict_period:
                impulse_matrix[j, :, i] = t_impulse
                a_matrix[:, j] = a
                i += 1
            else:
                a = -a  # variance control method

                if i < restrict_period and (t_impulse[larger_restrict_array] > 0.0).all() and (
                        t_impulse[less_restrict_array] < 0.0).all():
                    impulse_matrix[j, :, i] = t_impulse
                    a_matrix[:, j] = a
                    i += 1
                elif i >= restrict_period:
                    impulse_matrix[j, :, i] = t_impulse
                    a_matrix[:, j] = a
                    i += 1
                else:
                    i = 0
                    a = get_sphere_rand()

    if plot:
        print a_matrix.mean(0).shape
        fig = plt.figure(figsize=(20, 10))
        plt.plot(impulse_matrix.mean(0)[0, :], label="SRB")
        plt.plot(impulse_matrix.mean(0)[1, :], label="DXI")
        plt.plot(impulse_matrix.mean(0)[2, :], label="DXJ")
        plt.plot(impulse_matrix.mean(0)[3, :], label="DJM")
        plt.plot(impulse_matrix.mean(0)[4, :], label="ORC58")
        plt.plot(impulse_matrix.mean(0)[5, :], label="ORC62")
        plt.legend()
        plt.show()

    return a_matrix


def rolling_estimate(args):
    data, idx, larger_constraints, less_constraints = args

    fit_lag = 100
    result_lst = []

    for i in xrange(len(data)):
        if i < fit_lag:
            continue
        print i
        this_data = data.iloc[i - fit_lag: i, :]
        var = sm.VAR(endog=this_data.values)
        result = var.fit(1, trend="nc")

        a_matrix = irf(result, idx, larger_constraints, less_constraints, 2, 200, 2, plot=False)
        result_lst.append(a_matrix.mean(1))

    result_ar = np.array(result_lst)

    fig = plt.figure(figsize=(20, 10))
    plt.plot(result_ar[:, 0], label="0")
    plt.plot(result_ar[:, 1], label="1")
    plt.plot(result_ar[:, 2], label="2")
    plt.plot(result_ar[:, 3], label="3")
    plt.plot(result_ar[:, 4], label="4")
    plt.plot(result_ar[:, 5], label="5")
    plt.legend()
    plt.savefig("{}_100.png".format(idx))

    with open(str(idx) + "_100", "wb") as fp:
        cp.dump(result_ar, fp)


if __name__ == "__main__":
    import multiprocessing as mp

    data = pd.read_csv("../test_data3.csv")

    la_constraints = np.array(([True, True, True, False], [False, False, True, False], [False, True, True, False], [False, False, True, False], [False, False, False, True], [False, False, False, True]))
    le_constraints = np.array(([False, False, False, False], [True, False, False, False], [True, False, False, False], [True, True, False, False], [True, False, False, False], [True, False, False, False]))

    arg_lst = [(data, i, la_constraints, le_constraints) for i in xrange(4)]

    pool = mp.Pool(4)
    pool.map(rolling_estimate, arg_lst)
    pool.close()
    pool.join()
    #a = irf(result, 0, larger_constraints, less_constraints, 5, 200, 2, plot=False)
    #print a
    # 钢铁限产
