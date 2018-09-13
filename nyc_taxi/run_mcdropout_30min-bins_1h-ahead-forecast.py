
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import os.path
import numpy as np
import pandas as pd
import scipy.stats
import tensorflow as tf
import keras.backend as K
import statsmodels.formula.api as smf
from datetime import datetime
from sklearn import datasets, linear_model
from matplotlib import pyplot as plt
from matplotlib import cm
from keras.layers.core import Dense, Activation, Dropout, Lambda
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Conv3D, Conv2D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers import BatchNormalization, LeakyReLU
from keras.models import Sequential
from keras import regularizers
from keras.callbacks import ModelCheckpoint
import warnings
warnings.filterwarnings("ignore")

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (16, 10)

# prevent tensorflow from allocating the entire GPU memory at once
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)



# ---------------- Config params

STEPS_AHEAD = 1
NUM_LAGS = 10
N_EPOCHS = 100 
NUM_MC_SAMPLES = 100 

i_train = 2*24*90
i_val = i_train + 2*24*30
i_test = i_val + 2*24*60

quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.60, 0.70, 0.80, 0.90, 0.95]
quantile_pairs = [(0.05, 0.95), (0.10, 0.90), (0.20, 0.80), (0.30, 0.70), (0.40, 0.60)]



# ---------------- Load data

def read_data(filename):
    meta = []
    original_series = []
    detrended_series = []
    removed_trend = []
    f = open(filename)
    f.readline() # skip header
    for line in f:
        date,hour,minute,pickups,detrended_pickups,removed_trend_str,removed_std = line.strip().split(",")
        date = datetime.strptime(date, "%Y-%m-%d")
        meta.append((date,int(hour),int(minute)))
        original_series.append(float(pickups))
        detrended_series.append(float(detrended_pickups))
        removed_trend.append(float(removed_trend_str))
    f.close()
    original_series = pd.Series(original_series)
    detrended_series = pd.Series(detrended_series)
    removed_trend = np.array(removed_trend)
    removed_std = float(removed_std)
    assert len(original_series) == len(detrended_series)
    assert len(original_series) == len(removed_trend)
    
    return original_series, detrended_series, removed_trend, removed_std

print "\nLoading data..."
nx = 12
ny = 12
matrix_series = np.zeros((8735,nx,ny))
matrix_detrended = np.zeros((8735,nx,ny))
matrix_trend = np.zeros((8735,nx,ny))
matrix_std = np.zeros((nx,ny))
i = 0
for x in np.arange(40.700,40.820,0.01):
    j = ny-1
    for y in np.arange(-74.020,-73.900,0.01):
        filename = "/mnt/sdb1/nyc_taxi_data/by_grid/10grid_30min_detrended/pickups_10_%.3f_%.3f_30min.csv" % (x,y)
        original_series, detrended_series, removed_trend, removed_std = read_data(filename)
        matrix_series[:,i,j] = original_series
        matrix_detrended[:,i,j] = detrended_series
        matrix_trend[:,i,j] = removed_trend
        matrix_std[i,j] = removed_std
        j -= 1
    i += 1


# ---------------- Build lags

print "\nBuilding lags..."
matrix_lags = np.zeros((8735,NUM_LAGS,nx,ny))
for i in xrange(nx):
    for j in xrange(ny):
        lags = pd.concat([pd.Series(matrix_detrended[:,i,j]).shift(x) for x in range(1,NUM_LAGS+1)],axis=1)
        matrix_lags[:,:,i,j] = lags
        matrix_lags[:-NUM_LAGS,:,i,j] = matrix_lags[NUM_LAGS:,:,i,j]
        matrix_series[:-NUM_LAGS,i,j] = matrix_series[NUM_LAGS:,i,j]
        matrix_detrended[:-NUM_LAGS,i,j] = matrix_detrended[NUM_LAGS:,i,j]
        matrix_trend[:-NUM_LAGS,i,j] = matrix_trend[NUM_LAGS:,i,j]


# ---------------- Train/test split

X_train = np.zeros((i_train,NUM_LAGS,nx,ny))
y_train = np.zeros((i_train,nx,ny))
X_val = np.zeros((i_val-i_train,NUM_LAGS,nx,ny))
y_val = np.zeros((i_val-i_train,nx,ny))
X_test = np.zeros((i_test-i_val,NUM_LAGS,nx,ny))
y_test = np.zeros((i_test-i_val,nx,ny))
for i in xrange(nx):
    for j in xrange(ny):
        X_train[:,:,i,j] = matrix_lags[:i_train,:,i,j]
        y_train[:,i,j] = matrix_detrended[:i_train,i,j]
        X_val[:,:,i,j] = matrix_lags[i_train:i_val,:,i,j]
        y_val[:,i,j] = matrix_detrended[i_train:i_val,i,j]
        X_test[:,:,i,j] = matrix_lags[i_val:i_test,:,i,j]
        y_test[:,i,j] = matrix_detrended[i_val:i_test,i,j]

X_train = X_train[:,STEPS_AHEAD:,:,:]
X_val = X_val[:,STEPS_AHEAD:,:,:]
X_test = X_test[:,STEPS_AHEAD:,:,:]

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


def build_model(loss="mse", num_outputs=1):
    model = Sequential()

    model.add(ConvLSTM2D(filters=100, kernel_size=(3, 3),
                   input_shape=(None, nx, ny, 1),
                   padding='same', return_sequences=False))
    model.add(BatchNormalization())
    model.add(Lambda(lambda x: K.dropout(x, level=0.75)))

    model.add(Dense(units=num_outputs, 
                    kernel_regularizer=regularizers.l2(0.0001),
             ))
    model.add(Activation("linear"))

    model.compile(loss=loss, optimizer='nadam')
    
    return model


print "\nTraining NN for mean..."

# checkpoint best model
checkpoint = ModelCheckpoint("weights_mc.best.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# fit model to the mean
model_mc = build_model(loss="mse", num_outputs=1)
model_mc.fit(
    X_train[:,:,:,:,np.newaxis],
    y_train[:,:,:,np.newaxis],
    batch_size=128,
    epochs=N_EPOCHS, #100
    validation_data=(X_val[:,:,:,:,np.newaxis],y_val[:,:,:,np.newaxis]),
    callbacks=[checkpoint],
    verbose=0)   


# load weights
model_mc.load_weights("weights_mc.best.hdf5")

# make predictions
predictions = np.zeros((NUM_MC_SAMPLES, len(X_test), nx, ny))
for s in xrange(NUM_MC_SAMPLES):
    preds = model_mc.predict(X_test[:,:,:,:,np.newaxis])
    predictions[s,:,:,:] = preds[:,:,:,0]
    rmse = np.sqrt(np.mean(preds[:,:,:,0] - y_test[:,:,:])**2)
    print "[%d] test rmse=%.4f" % (s, rmse)

# make predictions
trues_sigma = y_val[:,:,:]
predictions_sigma = np.zeros((NUM_MC_SAMPLES, len(X_val), nx, ny))
for s in xrange(NUM_MC_SAMPLES):
    preds = model_mc.predict(X_val[:,:,:,:,np.newaxis])
    predictions_sigma[s,:,:,:] = preds[:,:,:,0]
    rmse = np.sqrt(np.mean(preds[:,:,:,0] - y_val[:,:,:])**2)
    print "[%d] val rmse=%.4f" % (s, rmse)

# select optimal value of alpha based on log probability on validation set
alphas = np.zeros((nx,ny))
for i in xrange(nx):
    for j in xrange(ny):
        means = np.mean(predictions_sigma[:,:,i,j], axis=0)
        vars_ = np.var(predictions_sigma[:,:,i,j], axis=0)
        
        best_log_prob = -np.inf
        for alpha in np.arange(0.0,2.0,0.01):
            log_prob = 0.0
            for n in xrange(len(y_val)):
                log_prob += np.log(scipy.stats.norm.pdf(trues_sigma[n,i,j], loc=means[n], scale=np.sqrt(vars_[n]+alpha)))
            if log_prob > best_log_prob:
                best_log_prob = log_prob
                alphas[i,j] = alpha
        print "optimal alpha for place (%d,%d): %.3f -> log_prob=%.3f" % (i,j,alphas[i,j],best_log_prob)

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
for i in xrange(nx):
    for j in xrange(ny):
        trues = y_test[:,i,j] * matrix_std[i,j] + matrix_trend[i_val:i_test,i,j]
        preds_mc = np.mean(predictions[:,:,i,j], axis=0) * matrix_std[i,j] + matrix_trend[i_val:i_test,i,j]
        
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
for i in xrange(nx):
    for j in xrange(ny):
        for k in xrange(len(quantiles)):
            trues = y_test[:,i,j] * matrix_std[i,j] + matrix_trend[i_val:i_test,i,j]
            preds_mc = np.mean(predictions[:,:,i,j], axis=0) * matrix_std[i,j] + matrix_trend[i_val:i_test,i,j]
            
            sigma2_est = np.mean((np.mean(predictions_sigma, axis=0) - trues_sigma)**2, axis=0)
            var_dl = np.var(predictions[:,:,i,j], axis=0) 
            
            inv_cdf = scipy.stats.norm.ppf(quantiles[k])
            # here we use different alternatives for calibrating the uncertainty estimates produced by MCdropout, explained in the paper
            #quant_mc = (np.mean(predictions, axis=0)[:,i,j] + inv_cdf*np.sqrt(var_dl)) * matrix_std[i,j] + matrix_trend[i_val:i_test,i,j]
            quant_mc = (np.mean(predictions, axis=0)[:,i,j] + inv_cdf*np.sqrt(var_dl + sigma2_est[i,j])) * matrix_std[i,j] + matrix_trend[i_val:i_test,i,j]
            #quant_mc = (np.mean(predictions, axis=0)[:,i,j] + inv_cdf*np.sqrt(var_dl + alphas[i,j])) * matrix_std[i,j] + matrix_trend[i_val:i_test,i,j]
            #quant_mc = np.percentile(predictions, quantiles[k]*100, axis=0)[:,i,j] * matrix_std[i,j] + matrix_trend[i_val:i_test,i,j]

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
    for i in xrange(nx):
        for j in xrange(ny):
            trues = y_test[:,i,j] * matrix_std[i,j] + matrix_trend[i_val:i_test,i,j]
            preds_mc = np.mean(predictions[:,:,i,j], axis=0) * matrix_std[i,j] + matrix_trend[i_val:i_test,i,j]

            sigma2_est = np.mean((np.mean(predictions_sigma, axis=0) - trues_sigma)**2, axis=0)
            var_dl = np.var(predictions[:,:,i,j], axis=0) 
            
            inv_cdf_lower = scipy.stats.norm.ppf(q_low)
            # here we use different alternatives for calibrating the uncertainty estimates produced by MCdropout, explained in the paper
            #lower_mc = (np.mean(predictions, axis=0)[:,i,j] + inv_cdf_lower*np.sqrt(var_dl)) * matrix_std[i,j] + matrix_trend[i_val:i_test,i,j]
            lower_mc = (np.mean(predictions, axis=0)[:,i,j] + inv_cdf_lower*np.sqrt(var_dl + sigma2_est[i,j])) * matrix_std[i,j] + matrix_trend[i_val:i_test,i,j]
            #lower_mc = (np.mean(predictions, axis=0)[:,i,j] + inv_cdf_lower*np.sqrt(var_dl + alphas[i,j])) * matrix_std[i,j] + matrix_trend[i_val:i_test,i,j]
            #lower_mc = np.percentile(predictions, q_low*100, axis=0)[:,i,j] * matrix_std[i,j] + matrix_trend[i_val:i_test,i,j]

            inv_cdf_upper = scipy.stats.norm.ppf(q_high)
            # here we use different alternatives for calibrating the uncertainty estimates produced by MCdropout, explained in the paper
            #upper_mc = (np.mean(predictions, axis=0)[:,i,j] + inv_cdf_upper*np.sqrt(var_dl)) * matrix_std[i,j] + matrix_trend[i_val:i_test,i,j]
            upper_mc = (np.mean(predictions, axis=0)[:,i,j] + inv_cdf_upper*np.sqrt(var_dl + sigma2_est[i,j])) * matrix_std[i,j] + matrix_trend[i_val:i_test,i,j]
            #upper_mc = (np.mean(predictions, axis=0)[:,i,j] + inv_cdf_upper*np.sqrt(var_dl + alphas[i,j])) * matrix_std[i,j] + matrix_trend[i_val:i_test,i,j]
            #upper_mc = np.percentile(predictions, q_high*100, axis=0)[:,i,j] * matrix_std[i,j] + matrix_trend[i_val:i_test,i,j]

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

