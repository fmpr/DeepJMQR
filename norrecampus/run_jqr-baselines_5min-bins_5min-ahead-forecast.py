

import os
import time
import warnings
import numpy as np
#import tensorflow as tf
import pandas as pd
import keras.backend as K
import statsmodels.formula.api as smf
from sklearn import datasets, linear_model
from numpy import newaxis
from matplotlib import pyplot as plt
warnings.filterwarnings("ignore")

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (16, 10)

# prevent tensorflow from allocating the entire GPU memory at once
#config = tf.ConfigProto()
#config.gpu_options.allow_growth=True
#sess = tf.Session(config=config)

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
#warnings.filterwarnings("ignore") #Hide messy Numpy warnings


# ---------------- Config params

STEPS_AHEAD = 1
NUM_LAGS = 10

bootstrap_size = 12*24*1
i_train = 12*24*89
i_val = i_train + 12*24*30
i_test = i_val + 12*24*60
num_train = 3000 # subsample of dataset to use for training

quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.60, 0.70, 0.80, 0.90, 0.95]
quantile_pairs = [(0.05, 0.95), (0.10, 0.90), (0.20, 0.80), (0.30, 0.70), (0.40, 0.60)]

best_parameters = [(0.001, 0.001, 2.8678486652388), (0.1, 0.01, 3.0886326415388723), (0.1, 0.01, 2.8759364185787062), (0.1, 0.01, 3.0887319035996246), (0.001, 0.001, 2.8950382405243418), (0.1, 0.01, 3.0487102039618326), (0.1, 0.01, 2.9052815861454513), (0.1, 0.01, 2.7418270523345898), (0.1, 0.01, 3.0687695983775694)]

# ---------------- Load data

def load_data(filename, num_lags, bootstrap_size):
    assert bootstrap_size > num_lags
    
    # read data from file
    f = open(filename)
    series = []
    removed_seasonality = [] 
    removed_std = []
    missings = []
    for line in f:
        splt = line.split(",")
        series.append(float(splt[0]))
        removed_seasonality.append(float(splt[1]))
        removed_std.append(float(splt[2]))
        missings.append(int(splt[3]))
    series = np.array(series)
    removed_seasonality = np.array(removed_seasonality)
    removed_std = np.array(removed_std)
    missings = np.array(missings)
    f.close()

    # generate lags
    X = []
    for i in range(bootstrap_size, len(series)):
        X.append(series[i-num_lags:i])
    X = np.array(X)
    
    y = series[bootstrap_size:]
    removed_seasonality = removed_seasonality[bootstrap_size:]
    removed_std = removed_std[bootstrap_size:]
    missings = missings[bootstrap_size:]
    assert X.shape[0] == y.shape[0]

    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, removed_seasonality, removed_std, missings


# read data from different places
print "\nLoading data..."
ids = []
removed_seasonality_all = []
removed_std_all = []
missings_all = []
X_all = []
y_all = []
for fname in os.listdir('../norrecampus'):
    print "reading data for place id:", fname[-6:-4]
    X, y, removed_seasonality, removed_std, missings = load_data('../norrecampus/'+fname, NUM_LAGS, bootstrap_size)
    ids.append(fname[-6:-4])
    X_all.append(X)
    y_all.append(y)
    removed_seasonality_all.append(removed_seasonality)
    removed_std_all.append(removed_std)
    missings_all.append(missings)
X_all = np.array(X_all)
y_all = np.array(y_all)
n_places = len(ids)
n_instances = X_all.shape[1]
print "n_instances:", n_instances
print "n_places:", n_places
print "n_lags:", NUM_LAGS

# reshape data
X = np.swapaxes(X_all, 0, 1)
X = np.swapaxes(X, 1, 2)
X = X[:,:,:,:,np.newaxis]
y = np.swapaxes(y_all, 0, 1)
y = y[:,np.newaxis,:,np.newaxis,np.newaxis]

X = X[:,:NUM_LAGS-STEPS_AHEAD+1,:,:]



# ---------------- Train/test split

X_train = X[:i_train,:]
y_train = y[:i_train]
X_val = X[i_train:i_val,:]
y_val = y[i_train:i_val]
X_test = X[i_val:i_test,:]
y_test = y[i_val:i_test]
print X_train.shape
print y_train.shape
print X_val.shape
print y_val.shape
print X_test.shape
print y_test.shape




# ---------------- Evaluation functions

def compute_error(trues, predicted):
    corr = np.corrcoef(predicted, trues)[0,1]
    mae = np.mean(np.abs(predicted - trues))
    mse = np.mean((predicted - trues)**2)
    rae = np.sum(np.abs(predicted - trues)) / np.sum(np.abs(trues - np.mean(trues)))
    rmse = np.sqrt(np.mean((predicted - trues)**2))
    r2 = max(0, 1 - np.sum((trues-predicted)**2) / np.sum((trues - np.mean(trues))**2))
    return corr, mae, mse, rae, rmse, r2


# In[7]:

def compute_error_filtered(trues, predicted, filt):
    trues = trues[filt]
    predicted = predicted[filt]
    corr = np.corrcoef(predicted, trues)[0,1]
    mae = np.mean(np.abs(predicted - trues))
    mse = np.mean((predicted - trues)**2)
    rae = np.sum(np.abs(predicted - trues)) / np.sum(np.abs(trues - np.mean(trues)))
    rmse = np.sqrt(np.mean((predicted - trues)**2))
    r2 = max(0, 1 - np.sum((trues-predicted)**2) / np.sum((trues - np.mean(trues))**2))
    return corr, mae, mse, rae, rmse, r2


# In[8]:

def eval_quantiles(lower, upper, trues, preds):
    N = len(trues)
    icp = 1.0*np.sum((trues>lower) & (trues<upper)) / N
    diffs = np.maximum(0, upper-lower)
    mil = np.sum(diffs) / N
    rmil = 0.0
    for i in xrange(N):
        if trues[i] != preds[i]:
            rmil += diffs[i] / (np.abs(trues[i]-preds[i]))
    rmil = rmil / N
    clc = np.exp(-rmil*(icp-0.95))
    return icp, mil, rmil, clc

# ---------------- Linear regression baseline

print "running linear regression baselines..."

col_names = ["l%d" % (i,) for i in xrange(NUM_LAGS-STEPS_AHEAD+1)]
col_names.append("y")
qr_str = "y ~ " + ''.join(["+l%d" % (x,) for x in range(NUM_LAGS-STEPS_AHEAD+1)])[1:]

predictions_lr = []
for k in xrange(len(quantiles)):
    q = quantiles[k]
    print "running quantile regression for quantile:", q
    
    preds_lr = np.zeros((i_test-i_val, n_places))
    for i in xrange(n_places):
        regr = linear_model.LinearRegression()
        regr.fit(X_train[:,:,i,0,0], y_train[:,0,i,0,0])
        preds_lr[:,i] = regr.predict(X_test[:,:,i,0,0]) * removed_std_all[i][i_val:i_test] + removed_seasonality_all[i][i_val:i_test]

    predictions_lr.append(preds_lr)


# ---------------- Joint Quantile Regression in RKHS

from operalib import Quantile

def tilted_loss_np(q,y,f):
    e = (y-f)
    # The term inside k.mean is a one line simplification of the first equation
    return np.mean(q*e + np.clip(-e, K.epsilon(), np.inf), axis=-1)

rand_ix = np.random.permutation(X_train.shape[0])
rand_ix = rand_ix[:num_train]

sum_mae = 0.0
sum_rmse = 0.0
sum_r2 = 0.0
sum_icp = [0.0 for j in quantile_pairs]
sum_mil = [0.0 for j in quantile_pairs]
sum_rmil = [0.0 for j in quantile_pairs]
sum_tloss = 0.0
sum_closs = 0.0
total_crosses = 0
for ix in xrange(n_places):
    print "running JQR for place id:", ids[ix] 

    #jqr_model = Quantile(probs=quantiles, kernel='DGauss', lbda=best_parameters[ix][0], gamma=best_parameters[ix][1], gamma_quantile=np.inf)
    jqr_model = Quantile(probs=quantiles, kernel='DGauss', lbda=best_parameters[ix][0], gamma=best_parameters[ix][1], gamma_quantile=1e-3)

    jqr_model.fit(X_train[rand_ix,:,ix,0,0], y_train[rand_ix,0,ix,0,0])

    preds = jqr_model.predict(X_test[:,:,ix,0,0])
    
    trues = y_test[:,0,ix,0,0] * removed_std_all[ix][i_val:i_test] + removed_seasonality_all[ix][i_val:i_test]
    preds_qr = predictions_lr[0][:,ix] 

    corr, mae, mse, rae, rmse, r2 = compute_error(trues, preds_qr)
    sum_mae += mae
    sum_rmse += rmse
    sum_r2 += r2

    for j in xrange(len(quantile_pairs)):
        pair = quantile_pairs[j]
        q_low, q_high = pair
        q_low_ix = quantiles.index(q_low)
        q_high_ix = quantiles.index(q_high)

        lower_qr = preds[q_low_ix,:] * removed_std_all[ix][i_val:i_test] + removed_seasonality_all[ix][i_val:i_test]
        upper_qr = preds[q_high_ix,:] * removed_std_all[ix][i_val:i_test] + removed_seasonality_all[ix][i_val:i_test]

        icp, mil, rmil, clc = eval_quantiles(lower_qr, upper_qr, trues, preds_qr)
        print pair, "\t(JQR: %.3f %.3f %.3f)" % (icp, mil, rmil)
        sum_icp[j] += icp
        sum_mil[j] += mil
        sum_rmil[j] += rmil

    # tilted loss
    tloss = 0.0
    for k in xrange(len(quantiles)):
        quant_qr = preds[k,:] * removed_std_all[ix][i_val:i_test] + removed_seasonality_all[ix][i_val:i_test]
        tloss += tilted_loss_np(quantiles[k],trues,quant_qr)
    print "tilted loss:", tloss
    sum_tloss += tloss

    # cross loss and total crosses
    closs = 0.0
    for k in xrange(len(quantiles)-1):
        q1 = preds[k,:]
        q2 = preds[k+1,:]
        total_crosses += np.sum(q1 > q2)
        closs += np.sum(np.maximum(0.0, q1 - q2)) / len(q1)
    print "cross loss:", closs
    sum_closs += closs


fname = "results_jqr_%dlags" % (NUM_LAGS,)
for q in quantiles:
    fname += "_q%2d" % (int(q*100),)
fname = fname.replace(" ","0") + ".csv"
if not os.path.exists(fname):
    f_res = open(fname, "a")
    f_res.write("MAE_JQR,RMSE_JQR")
    f_res.write(",C_LOSS_JQR,CROSSES_JQR,T_LOSS_JQR")
    for p in quantile_pairs:
        f_res.write(",ICP_JQR,MIL_JQR")
    f_res.close()
f_res = open(fname, "a")
f_res.write("\n")

f_res.write("%.3f,%3f," % (sum_mae/n_places, sum_rmse/n_places))
f_res.write("%.3f,%d,%.3f" % (sum_closs, total_crosses, sum_tloss))
for j in xrange(len(sum_icp)):
    f_res.write(",%.3f,%.3f" % (sum_icp[j]/n_places, sum_mil[j]/n_places))

f_res.close()



