import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.vector_ar import var_model as sm
import numpy as np
from math import sqrt


def get_sphere_rand():
    trand = np.random.rand(len(sigma)) - 0.5
    return trand / sqrt(sum(trand**2))


data = pd.read_csv("../test_data.csv", index_col=0)
var = sm.VAR(endog=data.values)

result = var.fit(maxlags=1)
A_1 = result.coefs[0, :, :]
sigma = result.sigma_u

P = np.linalg.cholesky(sigma)
diag = np.diag(P)
D = np.eye(len(P)) * P
A = D.dot(np.linalg.inv(P))

MAX_COUNT = 2e5

a = get_sphere_rand()
count = 0
larger_constraints = np.array([0.0, 0.0, np.nan, np.nan])
less_constraints = np.array([np.nan, np.nan, 0.0, 0.0])

while count < MAX_COUNT:
    a = get_sphere_rand()
    t_impulse = A.dot(a)
    if (t_impulse > larger_constraints).all() or (t_impulse < less_constraints).all():
        print t_impulse
        break