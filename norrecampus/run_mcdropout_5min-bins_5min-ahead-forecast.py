
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

import os
import time
import warnings
import numpy as np
import tensorflow as tf
import pandas as pd
import scipy.stats
import keras.backend as K
import statsmodels.formula.api as smf
from sklearn import datasets, linear_model
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout, Lambda
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers import BatchNormalization
from keras.models import Sequential
from keras import regularizers
from matplotlib import pyplot as plt
from keras.callbacks import ModelCheckpoint
warnings.filterwarnings("ignore")

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (16, 10)

# prevent tensorflow from allocating the entire GPU memory at once
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
#warnings.filterwarnings("ignore") #Hide messy Numpy warnings


# ---------------- Config params

STEPS_AHEAD = 1
NUM_LAGS = 10
N_EPOCHS = 100 
NUM_MC_SAMPLES = 100 

bootstrap_size = 12*24*1
i_train = 12*24*89
i_val = i_train + 12*24*30
i_test = i_val + 12*24*60

quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.60, 0.70, 0.80, 0.90, 0.95]
quantile_pairs = [(0.05, 0.95), (0.10, 0.90), (0.20, 0.80), (0.30, 0.70), (0.40, 0.60)]



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



# ---------------- Evaluation functions

def compute_error(trues, predicted):
    corr = np.corrcoef(predicted, trues)[0,1]
    mae = np.mean(np.abs(predicted - trues))
    mse = np.mean((predicted - trues)**2)
    rae = np.sum(np.abs(predicted - trues)) / np.sum(np.abs(trues - np.mean(trues)))
    rmse = np.sqrt(np.mean((predicted - trues)**2))
    r2 = max(0, 1 - np.sum((trues-predicted)**2) / np.sum((trues - np.mean(trues))**2))
    return corr, mae, mse, rae, rmse, r2


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



# ---------------- Independent deep learning models for mean and quantiles

def tilted_loss(q,y,f):
    e = (y-f)
    # The term inside k.mean is a one line simplification of the first equation
    return K.mean(q*e + K.clip(-e, K.epsilon(), np.inf), axis=-1)


def build_model(loss="mse", num_outs=1):
    model = Sequential()

    model.add(ConvLSTM2D(filters=20, kernel_size=(3, 3),
                   input_shape=(None, 9, 1, 1),
                   padding='same', return_sequences=True))
    model.add(Lambda(lambda x: K.dropout(x, level=0.2)))

    model.add(ConvLSTM2D(filters=20, kernel_size=(3, 3),
                       padding='same', return_sequences=False))
    model.add(Lambda(lambda x: K.dropout(x, level=0.2)))
    
    model.add(Dense(units=num_outs))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss=loss, optimizer="rmsprop")
    
    return model


print "\nTraining NN for mean..."

model = build_model(num_outs=1)

# checkpoint best model
checkpoint = ModelCheckpoint("convlstm_mc.best.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')

model.fit(
    X_train,
    y_train[:,:,:,:,0].swapaxes(1,2),
    batch_size=512,
    epochs=N_EPOCHS,
    validation_data=(X_val,y_val[:,:,:,:,0].swapaxes(1,2)),
    shuffle=True,
    callbacks=[checkpoint],
    verbose=2)  

# load weights
model.load_weights("convlstm_mc.best.hdf5")

# make predictions
predictions = np.zeros((NUM_MC_SAMPLES, len(X_test), 9))
for s in xrange(NUM_MC_SAMPLES):
    preds = model.predict(X_test)
    predictions[s,:,:] = preds[:,:,0,0]
    rmse = np.sqrt(np.mean(preds[:,:,0,0] - y_test[:,0,:,0,0])**2)
    print "[%d] test rmse=%.4f" % (s, rmse)

# make predictions
trues_sigma = y_val[:,0,:,0,0]
predictions_sigma = np.zeros((NUM_MC_SAMPLES, len(X_val), 9))
for s in xrange(NUM_MC_SAMPLES):
    preds = model.predict(X_val)
    predictions_sigma[s,:,:] = preds[:,:,0,0]
    rmse = np.sqrt(np.mean(preds[:,:,0,0] - y_val[:,0,:,0,0])**2)
    print "[%d] val rmse=%.4f" % (s, rmse)

# select optimal value of alpha based on log probability on validation set
alphas = np.zeros((n_places,))
for i in xrange(n_places):
    means = np.mean(predictions_sigma[:,:,i], axis=0)
    vars_ = np.var(predictions_sigma[:,:,i], axis=0)
    
    print "finding optimal alpha for place:", i
    best_log_prob = -np.inf
    for alpha in np.arange(0.0,1.0,0.01):
        log_prob = 0.0
        for n in xrange(len(y_val)):
            log_prob += np.log(scipy.stats.norm.pdf(trues_sigma[n,i], loc=means[n], scale=np.sqrt(vars_[n]+alpha)))
        if log_prob > best_log_prob:
            best_log_prob = log_prob
            alphas[i] = alpha

print "\nBest alphas:"
print alphas


# ---------------- Evaluate different methods

print "\nEvaluating different methods..."

fname = "results_mcdropout_%dlags" % (NUM_LAGS,)
for q in quantiles:
    fname += "_q%2d" % (int(q*100),)
fname = fname.replace(" ","0") + ".csv"
if not os.path.exists(fname):
    f_res = open(fname, "a")
    f_res.write("MAE_MC,RMSE_MC")
    f_res.write(",C_LOSS_MC,CROSSES_MC,T_LOSS_MC")
    for p in quantile_pairs:
        f_res.write(",ICP_MC,MIL_MC")
    f_res.close()
f_res = open(fname, "a")
f_res.write("\n")


# evaluate mean prediction

maes_mc = []
rmses_mc = []
r2s_mc = []
for i in xrange(n_places):
    trues = y_test[:,0,i,0,0] * removed_std_all[i][i_val:i_test] + removed_seasonality_all[i][i_val:i_test]
    preds_mc = np.mean(predictions, axis=0)[:,i] * removed_std_all[i][i_val:i_test] + removed_seasonality_all[i][i_val:i_test]
    
    # evaluate prediction means
    s = "[%d]\tMean:" % (i,)
    corr, mae, mse, rae, rmse, r2 = compute_error(trues, preds_mc)
    maes_mc.append(mae)
    rmses_mc.append(rmse)
    r2s_mc.append(r2)
    s += "\t(MC:    %.2f %.2f %.2f)" % (mae,rmse,r2)
    #print s

print "ERRORS:"
print "MC:    %.3f %.3f %.3f" % (np.mean(maes_mc),np.mean(rmses_mc),np.mean(r2s_mc)) 
f_res.write("%.3f,%.3f," % (np.mean(maes_mc),np.mean(rmses_mc)))


# evaluate crossing quantiles

loss_mc = 0.0 # for MCdropout is always zero
print "CROSS LOSS MC:", loss_mc
f_res.write("%.3f," % (loss_mc,))


sum_mc = 0 # for MCdropout is always zero
print "NUM CROSSES MC:", sum_mc
f_res.write("%d," % (sum_mc,))


# evaluate tilted losses

def tilted_loss_np(q,y,f):
    e = (y-f)
    # The term inside k.mean is a one line simplification of the first equation
    return np.mean(q*e + np.clip(-e, K.epsilon(), np.inf), axis=-1)

sum_mc = 0
for i in xrange(n_places):
    for k in xrange(len(quantiles)):
        trues = y_test[:,0,i,0,0] * removed_std_all[i][i_val:i_test] + removed_seasonality_all[i][i_val:i_test]

        preds_mc = np.mean(predictions, axis=0)[:,i] * removed_std_all[i][i_val:i_test] + removed_seasonality_all[i][i_val:i_test]
        
        sigma2_est = np.mean((np.mean(predictions_sigma, axis=0) - trues_sigma)**2, axis=0)
        var_dl = np.var(predictions, axis=0)[:,i] 
        
        inv_cdf = scipy.stats.norm.ppf(quantiles[k])
        # here we use different alternatives for calibrating the uncertainty estimates produced by MCdropout, explained in the paper
        quant_mc = (np.mean(predictions, axis=0)[:,i] + inv_cdf*np.sqrt(var_dl)) * removed_std_all[i][i_val:i_test] + removed_seasonality_all[i][i_val:i_test]
        #quant_mc = (np.mean(predictions, axis=0)[:,i] + inv_cdf*np.sqrt(var_dl + sigma2_est[i])) * removed_std_all[i][i_val:i_test] + removed_seasonality_all[i][i_val:i_test]
        #quant_mc = (np.mean(predictions, axis=0)[:,i] + inv_cdf*np.sqrt(var_dl + alphas[i])) * removed_std_all[i][i_val:i_test] + removed_seasonality_all[i][i_val:i_test]
        #quant_mc = np.percentile(predictions, quantiles[k]*100, axis=0)[:,i] * removed_std_all[i][i_val:i_test] + removed_seasonality_all[i][i_val:i_test]

        sum_mc += tilted_loss_np(quantiles[k],trues,quant_mc)

print "TILTED LOSS MC:", sum_mc
f_res.write("%.3f," % (sum_mc,))


# evaluate quantiles

for pair in quantile_pairs:
    q_low, q_high = pair
    q_low_ix = quantiles.index(q_low)
    q_high_ix = quantiles.index(q_high)

    icps_mc = []
    mils_mc = []
    rmils_mc = []
    for i in xrange(n_places):
        trues = y_test[:,0,i,0,0] * removed_std_all[i][i_val:i_test] + removed_seasonality_all[i][i_val:i_test]

        preds_mc = np.mean(predictions, axis=0)[:,i] * removed_std_all[i][i_val:i_test] + removed_seasonality_all[i][i_val:i_test]
        
        sigma2_est = np.mean((np.mean(predictions_sigma, axis=0) - trues_sigma)**2, axis=0)
        var_dl = np.var(predictions, axis=0)[:,i] 
        
        inv_cdf_lower = scipy.stats.norm.ppf(q_low)
        # here we use different alternatives for calibrating the uncertainty estimates produced by MCdropout, explained in the paper
        lower_mc = (np.mean(predictions, axis=0)[:,i] + inv_cdf_lower*np.sqrt(var_dl)) * removed_std_all[i][i_val:i_test] + removed_seasonality_all[i][i_val:i_test]
        #lower_mc = (np.mean(predictions, axis=0)[:,i] + inv_cdf_lower*np.sqrt(var_dl + sigma2_est[i])) * removed_std_all[i][i_val:i_test] + removed_seasonality_all[i][i_val:i_test]
        #lower_mc = (np.mean(predictions, axis=0)[:,i] + inv_cdf_lower*np.sqrt(var_dl + alphas[i])) * removed_std_all[i][i_val:i_test] + removed_seasonality_all[i][i_val:i_test]
        #lower_mc = np.percentile(predictions, q_low*100, axis=0)[:,i] * removed_std_all[i][i_val:i_test] + removed_seasonality_all[i][i_val:i_test]
        
        inv_cdf_upper = scipy.stats.norm.ppf(q_high)
        # here we use different alternatives for calibrating the uncertainty estimates produced by MCdropout, explained in the paper
        upper_mc = (np.mean(predictions, axis=0)[:,i] + inv_cdf_upper*np.sqrt(var_dl)) * removed_std_all[i][i_val:i_test] + removed_seasonality_all[i][i_val:i_test]
        #upper_mc = (np.mean(predictions, axis=0)[:,i] + inv_cdf_upper*np.sqrt(var_dl + sigma2_est[i])) * removed_std_all[i][i_val:i_test] + removed_seasonality_all[i][i_val:i_test]
        #upper_mc = (np.mean(predictions, axis=0)[:,i] + inv_cdf_upper*np.sqrt(var_dl + alphas[i])) * removed_std_all[i][i_val:i_test] + removed_seasonality_all[i][i_val:i_test]
        #upper_mc = np.percentile(predictions, q_high*100, axis=0)[:,i] * removed_std_all[i][i_val:i_test] + removed_seasonality_all[i][i_val:i_test]

        # evaluate quantiles
        s = "[%d]\tQuantiles:" % (i,)
        icp, mil, rmil, clc = eval_quantiles(lower_mc, upper_mc, trues, preds_mc)
        icps_mc.append(icp)
        mils_mc.append(mil)
        rmils_mc.append(rmil)
        s += "\t(MC:    %.2f %.2f %.2f)" % (icp, mil, rmil)
        #print s

    print "QUANTILES (MEAN):"
    print "MC:    %.3f %.3f %.3f" % (np.mean(icps_mc),np.mean(mils_mc),np.mean(rmils_mc)) 
    f_res.write("%.3f,%.3f," % (np.mean(icps_mc),np.mean(mils_mc)))

    print "QUANTILES (MEDIAN):"
    print "MC:    %.3f %.3f %.3f" % (np.median(icps_mc),np.median(mils_mc),np.median(rmils_mc)) 


f_res.close()


