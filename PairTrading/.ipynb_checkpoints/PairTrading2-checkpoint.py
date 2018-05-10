import pandas as pd
import numpy as np
import multiprocessing as mp
from math import sqrt
import cPickle as cp
import matplotlib.pyplot as plt
from arch.univariate import ARX, EGARCH, SkewStudent


def back_test_sharp(args): # at this version their is no stop loss
    s_score, resid_se, buy_stop_p, buy_open_p, buy_close_p, sell_stop_p, sell_open_p, sell_close_p = args
    buy_open = s_score.quantile(buy_open_p)
    buy_close = s_score.quantile(buy_close_p)
    sell_open = s_score.quantile(sell_open_p)
    sell_close = s_score.quantile(sell_close_p)
    buy_stop = s_score.quantile(buy_stop_p)
    sell_stop = s_score.quantile(sell_stop_p)
        
    signal_ar = np.zeros((len(resid_se), 1))
    
    for i in xrange(1, len(resid_se)):
        if s_score[i] < sell_open and s_score[i - 1] >= sell_open: # sell open
            signal_ar[i] = -1
        elif s_score[i] < sell_close and s_score[i - 1] >= sell_close: # sell close
            signal_ar[i] = 0
        elif s_score[i] > buy_open and s_score[i - 1] <= buy_open: # buy open
            signal_ar[i] = 1
        elif s_score[i] > buy_close and s_score[i - 1] <= buy_close: # buy close
            signal_ar[i] = 0
        elif (s_score[i] < buy_stop and s_score[i - 1] >= buy_stop) or (s_score[i] > sell_stop and s_score[i - 1] <= sell_stop):
        	signal_ar[i] = 0
        else: # do nothing
            signal_ar[i] = signal_ar[i - 1]
    
    return_rate = (resid_se[2:].values * signal_ar[:-2].T).cumsum()
    return (return_rate[-1] / return_rate.std(), buy_stop_p, buy_open_p, buy_close_p, sell_stop_p, sell_open_p, sell_close_p)


def back_test(args): # at this version their is no stop loss
    s_score, resid_se, buy_stop_p, buy_open_p, buy_close_p, sell_stop_p, sell_open_p, sell_close_p = args
    buy_open = s_score.quantile(buy_open_p)
    buy_close = s_score.quantile(buy_close_p)
    sell_open = s_score.quantile(sell_open_p)
    sell_close = s_score.quantile(sell_close_p)
    buy_stop = s_score.quantile(buy_stop_p)
    sell_stop = s_score.quantile(sell_stop_p)
        
    signal_ar = np.zeros((len(resid_se), 1))

    
    for i in xrange(1, len(resid_se)):
        if s_score[i] < sell_open and s_score[i - 1] >= sell_open: # sell open
            signal_ar[i] = -1
        elif s_score[i] < sell_close and s_score[i - 1] >= sell_close: # sell close
            signal_ar[i] = 0
        elif s_score[i] > buy_open and s_score[i - 1] <= buy_open: # buy open
            signal_ar[i] = 1
        elif s_score[i] > buy_close and s_score[i - 1] <= buy_close: # buy close
            signal_ar[i] = 0
        elif (s_score[i] < buy_stop and s_score[i - 1] >= buy_stop) or (s_score[i] > sell_stop and s_score[i - 1] <= sell_stop):
        	signal_ar[i] = 0
        else: # do nothing
            signal_ar[i] = signal_ar[i - 1]
    
    return ((resid_se[2:].values * signal_ar[:-2].T).cumsum(), buy_stop_p, buy_open_p, buy_close_p, sell_stop_p, sell_open_p, sell_close_p)


def main(ticker1, ticker2):
	df = pd.read_csv("./Data/close.csv", dtype={"date": str})

	df2 = np.log(df.loc[:, [ticker1, ticker2]]).diff().dropna()
	x = df2[ticker1].values
	y = df2[ticker2].values
	A = np.vstack((np.ones_like(x), x)).T

	b = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(y)
	resid = y - A.dot(b)

	resid_se = pd.Series(resid)
	std2_se = resid_se.rolling(
	    window=100,
	).apply(lambda x: sqrt(sum(np.diff(x)**2) / (len(x) - 1)))
	mean_se = resid_se.rolling(
	    window=100,
	).mean()

	'''
	s_score = (pd.Series(resid_se) - mean_se) / std2_se
	'''
	ar = ARX(resid_se, volatility=EGARCH(2, 0, 2))
	ar.distribution = SkewStudent()
	res = ar.fit()
	s_score = pd.Series(resid)

	arg_lst = [
		(s_score, resid_se, i / 100.0, j / 100.0, k / 100.0, l / 100.0, m / 100.0, n / 100.0) for i in xrange(15, 35, 5) for j in xrange(i + 1, 49, 5) for k in xrange(j + 1, 50, 5) for l in xrange(85, 65, -5) for m in xrange(l - 1, 51, -5) for n in xrange(m - 1, 50, -5)
		]

	pool = mp.Pool(6)
	result = pool.map(back_test_sharp, arg_lst)
	pool.close()
	pool.join()

	with open("./pkl/EG_result_lst_{}_{}_sharp".format(ticker1, ticker2), "wb") as fp:
		cp.dump(result, fp)
	
	x_mean = x.mean()
	y_mean = y.mean()
	pearson = (x - x_mean).dot(y - y_mean) / sqrt(sum((x - x_mean)**2)) / sqrt(sum((y - y_mean)**2))

	result.sort(key=lambda x: x[0], reverse=True)
	best = result[0]
	res = back_test((s_score, resid_se, best[1], best[2], best[3], best[4], best[5], best[6]))
	fig = plt.figure(figsize=(20, 10))
	plt.plot(res[0])
	plt.savefig("./Pics/net_value/EG_{}_{}.png".format(ticker1, ticker2))
	del fig
	return pd.Series(res[0]).to_csv("./xlsx/EG_{}_{}_{}_{}_{}_{}_{}_{}_{}.csv".format(ticker1, ticker2, pearson, best[1], best[2], best[3], best[4], best[5], best[6]))

if __name__ == "__main__":
	hs_class = [
    "SRB", "SHC", "DXI", "DXJ", "DJM"
	]

	yz_class = [
	    "DXM", "DXY", "DXP", "ZOI"
	]

	gjs_class = [
	    "SAU", "SAG"
	]

	bond_class = [
	    "FXT", "FTF"
	]

	stock_class = [
	    "FIF", "FIC", "FIH"
	]

	price_class = [
	    "open", "high", "close", "low", "settle"
	]

	for idx, t1 in enumerate(bond_class):
		for t2 in bond_class[idx:]:
			if t1 == t2:
				continue
			main(t1, t2)
	
	for idx, t1 in enumerate(stock_class):
		for t2 in stock_class[idx:]:
			if t1 == t2:
				continue
			main(t1, t2)

	for idx, t1 in enumerate(hs_class):
		for t2 in hs_class[idx:]:
			if t1 == t2:
				continue
			main(t1, t2)

	for idx, t1 in enumerate(yz_class):
		for t2 in yz_class[idx:]:
			if t1 == t2:
				continue
			main(t1, t2)

	for idx, t1 in enumerate(gjs_class):
		for t2 in gjs_class[idx:]:
			if t1 == t2:
				continue
			main(t1, t2)
