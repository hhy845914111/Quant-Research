import matplotlib.pyplot as plt
import pandas as pd

from statsmodels.tsa.stattools import coint as st_coint
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
import numpy as np

import statsmodels.api as sm
from scipy.optimize import basinhopping
from scipy.stats import norm

import cPickle as cp
from math import sqrt

