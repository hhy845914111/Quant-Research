import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import cPickle as cp

def date2int(date_obj):
    string = str(date_obj)
    return int("".join(string.split(" ")[0].split("-")))

def delete_zero(df):
    for i in df:
        df[i] = df[i].fillna(0)
        if (df[i] == 0).all():
            del df[i]
    return df

def date2int(date_obj):
    string = str(date_obj)
    return int("".join(string.split(" ")[0].split("-")))


def optimize(y, x, lmbda, learn_rate, criteria):
    w = np.random.rand(x.shape[1], 1)
    last_loss = 0.0
    loss = criteria + last_loss
    while abs(loss - last_loss) >= criteria:
        gradient = 2 * x.T.dot(x).dot(w) - 2 * x.T.dot(y) + 2 * lmbda * w
        w = w - gradient * learn_rate
        w[w < 0.0] = 0.0
        last_loss = loss
        loss = sum((y - x.dot(w))**2) + lmbda * w.T.dot(w)
    return w

df_cta = pd.read_excel("./Data/cta_raw.xlsx", sheet_name="cta").iloc[:, :2]
df_cta["date"] = df_cta["date"].apply(date2int)
df_cta["NPV"] = df_cta["NPV"].pct_change()
df_5_13 = pd.read_excel("./Data/cta_raw.xlsx", sheet_name="5-13")
df_10_20 = pd.read_excel("./Data/cta_raw.xlsx", sheet_name="10-20")
df_20_50 = pd.read_excel("./Data/cta_raw.xlsx", sheet_name="20-50")
df_50_100 = pd.read_excel("./Data/cta_raw.xlsx", sheet_name="50-100")


def prepare_data_and_plot(args):
    df, df_name = args 
    lmbda=1e-7
    learn_rate=1e-2
    stop_rate=1e-4
    lag=40
    
    df = delete_zero(df)
    df_concated = pd.merge(df_cta, df, on="date", how="left").dropna()
    df_concated.iloc[:, 2:] = df_concated.iloc[:, 2:] / 1e7
    
    date = df_concated.iloc[:, 0].values
    y = df_concated.iloc[:, 1].values
    y.shape = (-1, 1)
    x = df_concated.iloc[:, 2:].values
    w = np.zeros_like(x.T)
    
    for i in xrange(len(x)):
        if i < lag:
            continue
        w_tmp = optimize(y[i-lag : i], x[i-lag : i, :], lmbda, learn_rate, stop_rate)
        w[:, i-lag : i] = w_tmp
    
    y_estimate = np.diag(x.dot(w))
    
    fig2 = plt.figure(figsize=(20, 10))
    plt.plot(y_estimate.cumsum())
    fig2.savefig("./Pics/{}.png".format(df_name))
    
    fig1 = plt.figure(figsize=(20, 10))
    plt.plot(y.cumsum())
    fig1.savefig("./Pics/{}.png".format('y'))
    
    return df_name, w



if __name__ == "__main__":
	arg_lst = [(df_5_13, "df_5_13"), (df_10_20, "df_10_20"), (df_20_50, "df_20_50"), (df_50_100, "df_50_100")]
	pool = mp.Pool(4)
	result = pool.map(prepare_data_and_plot, arg_lst)

	with open("./Pkl/result", "wb") as fp:
		cp.dump(result, fp)

	pool.close()
	pool.join()
