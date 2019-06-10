
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import time
import warnings
import numpy as np
import tensorflow as tf
import pandas as pd
import keras.backend as K
import statsmodels.formula.api as smf
from sklearn import datasets, linear_model
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
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
N_EPOCHS_MULTI = 100 

bootstrap_size = 12*24*1
i_train = 12*24*89
i_val = i_train + 12*24*30
i_test =  i_val + 12*24*60

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

    outlier_prob = 0.05 
    for i in range(len(y)):
        if np.random.rand() < outlier_prob:
            # make data point an outlier by adding Gaussian noise
            y[i] += np.random.randn()*5 

    return X, y, removed_seasonality, removed_std, missings


# read data from different places
print "\nLoading data..."
ids = []
removed_seasonality_all = []
removed_std_all = []
missings_all = []
X_all = []
y_all = []
for fname in os.listdir('../../deep-norrecampus/norrecampus'):
    print "reading data for place id:", fname[-6:-4]
    X, y, removed_seasonality, removed_std, missings = load_data('../../deep-norrecampus/norrecampus/'+fname, NUM_LAGS, bootstrap_size)
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

print X.shape
print X_train.shape
print X_val.shape
print X_test.shape


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
    model.add(Dropout(0.2))

    model.add(ConvLSTM2D(filters=20, kernel_size=(3, 3),
                       padding='same', return_sequences=False))
    model.add(Dropout(0.2))
    
    model.add(Dense(units=num_outs))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss=loss, optimizer="rmsprop")
    
    return model


print "\nTraining NN for mean..."

model = build_model(num_outs=1)

# checkpoint best model
checkpoint = ModelCheckpoint("convlstm_mean.best.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')

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
model.load_weights("convlstm_mean.best.hdf5")

# make predictions
predictions = model.predict(X_test)


models_q = []
for q in quantiles:
    print "\nTraining NN for quantile %.2f..." % (q,)
    
    # checkpoint best model
    checkpoint = ModelCheckpoint("convlstm_q%2f.best.hdf5" % (q,), monitor='val_loss', verbose=0, save_best_only=True, mode='min')

    # fit model to the 5% quantile
    model_q = build_model(loss=lambda y,f: tilted_loss(q,y,f))
    model_q.fit(
        X_train,
        y_train[:,:,:,:,0].swapaxes(1,2),
        batch_size=512,
        epochs=N_EPOCHS,
        validation_data=(X_val,y_val[:,:,:,:,0].swapaxes(1,2)),
        shuffle=True,
        callbacks=[checkpoint],
        verbose=0)  
    models_q.append(model_q)


predictions_q = []
for i in xrange(len(quantiles)):
    q = quantiles[i]
    print "\nMaking predictions for quantile %.2f..." % (q,)
    
    # load weights
    models_q[i].load_weights("convlstm_q%2f.best.hdf5" % (q,))

    # make predictions
    predictions_q.append(models_q[i].predict(X_test))


# ---------------- Linear Quantile Regression baselines

col_names = ["l%d" % (i,) for i in xrange(NUM_LAGS-STEPS_AHEAD+1)]
col_names.append("y")
qr_str = "y ~ " + ''.join(["+l%d" % (x,) for x in range(NUM_LAGS-STEPS_AHEAD+1)])[1:]

predictions_lr = []
predictions_q_lr = []
for k in xrange(len(quantiles)):
    q = quantiles[k]
    print "\nTraining linear QR for quantile %.2f..." % (q,)
    
    preds_lr = np.zeros((i_test-i_val, n_places))
    preds_qr = np.zeros((i_test-i_val, n_places))
    for i in xrange(n_places):
        regr = linear_model.LinearRegression()
        regr.fit(X_train[:,:,i,0,0], y_train[:,0,i,0,0])

        preds_lr[:,i] = regr.predict(X_test[:,:,i,0,0]) * removed_std_all[i][i_val:i_test] + removed_seasonality_all[i][i_val:i_test]

        data = pd.DataFrame(data = np.hstack([X_train[:,:,i,0,0], y_train[:,0,i,0,:]]), columns = col_names)
        data_test = pd.DataFrame(data = np.hstack([X_test[:,:,i,0,0], y_test[:,0,i,0,:]]), columns = col_names)

        qr = smf.quantreg(qr_str, data)
        res = qr.fit(q=q)
        preds_qr[:,i] = res.predict(data_test) * removed_std_all[i][i_val:i_test] + removed_seasonality_all[i][i_val:i_test]
    
    predictions_lr.append(preds_lr)
    predictions_q_lr.append(preds_qr)


# ---------------- Simultaneous Mean and Quantiles NN

def multi_tilted_loss(quantiles,y,f):
    print y
    print f
    # The term inside k.mean is a one line simplification of the first equation
    loss = K.mean(K.square(y[:,:,:,0]-f[:,:,:,0]), axis=-1)
    for k in xrange(len(quantiles)):
        q = quantiles[k]
        e = (y[:,:,:,k+1]-f[:,:,:,k+1])
        loss += K.mean(q*e + K.clip(-e, K.epsilon(), np.inf), axis=-1)
    return loss


global previous_losses
previous_losses = [1.0 for q in xrange(len(quantiles)+1)]
def multi_tilted_loss_weighted(quantiles,y,f):
    global previous_losses
    max_prev_loss = tf.reduce_max(previous_losses)
    
    # The term inside k.mean is a one line simplification of the first equation
    losses = [(max_prev_loss/previous_losses[0]) * K.mean(K.square(y[:,:,:,0]-f[:,:,:,0]), axis=-1)]
    for k in xrange(len(quantiles)):
        q = quantiles[k]
        e = (y[:,:,:,k+1]-f[:,:,:,k+1])
        losses.append((max_prev_loss/previous_losses[k+1]) * K.mean(q*e + K.clip(-e, K.epsilon(), np.inf), axis=-1))
    previous_losses = losses
    loss = tf.reduce_sum(losses)
    return loss


y_traink = y_train[:,:,:,:,0].swapaxes(1,2)
for k in xrange(len(quantiles)):
    y_traink = np.concatenate((y_traink, y_train[:,:,:,:,0].swapaxes(1,2)), axis=3)

y_valk = y_val[:,:,:,:,0].swapaxes(1,2)
for k in xrange(len(quantiles)):
    y_valk = np.concatenate((y_valk, y_val[:,:,:,:,0].swapaxes(1,2)), axis=3)

print "\nTraining multi-quantile NN..."

# checkpoint best model
checkpoint = ModelCheckpoint("convlstm_multi.best.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# fit model to the 95% quantile
model_multi = build_model(loss=lambda y,f: multi_tilted_loss_weighted(quantiles,y,f), num_outs=1+len(quantiles))
model_multi.fit(
    X_train,
    y_traink,
    batch_size=128,
    epochs=N_EPOCHS_MULTI, # 100
    validation_data=(X_val,y_valk),
    shuffle=True,
    callbacks=[checkpoint],
    verbose=2)


# load weights
model_multi.load_weights("convlstm_multi.best.hdf5")

# make predictions
predictions_multi = model_multi.predict(X_test)


# ---------------- Evaluate different methods

print "\nEvaluating different methods..."

fname = "results_outliers_%dlags" % (NUM_LAGS,)
for q in quantiles:
    fname += "_q%2d" % (int(q*100),)
fname = fname.replace(" ","0") + ".csv"
if not os.path.exists(fname):
    f_res = open(fname, "a")
    f_res.write("MAE_QR,MAE_DL,MAE_MULTI,RMSE_QR,RMSE_DL,RMSE_MULTI")
    f_res.write(",C_LOSS_QR,C_LOSS_DL,C_LOSS_MULTI,CROSSES_QR,CROSSES_DL,CROSSES_MULTI,T_LOSS_QR,T_LOSS_DL,T_LOSS_MULTI")
    for p in quantile_pairs:
        f_res.write(",ICP_QR,ICP_DL,ICP_MULTI,MIL_QR,MIL_DL,MIL_MULTI")
    f_res.close()
f_res = open(fname, "a")
f_res.write("\n")


# evaluate mean prediction

maes_lr = []
rmses_lr = []
r2s_lr = []
maes_dl = []
rmses_dl = []
r2s_dl = []
maes_multi = []
rmses_multi = []
r2s_multi = []
for i in xrange(n_places):
    trues = y_test[:,0,i,0,0] * removed_std_all[i][i_val:i_test] + removed_seasonality_all[i][i_val:i_test]
    preds_qr = predictions_lr[0][:,i] 
    preds_dl = predictions[:,i,0,0] * removed_std_all[i][i_val:i_test] + removed_seasonality_all[i][i_val:i_test]
    preds_multi = predictions_multi[:,i,0,0] * removed_std_all[i][i_val:i_test] + removed_seasonality_all[i][i_val:i_test]

    # evaluate prediction means
    s = "[%d]\tMean:" % (i,)
    corr, mae, mse, rae, rmse, r2 = compute_error(trues, preds_qr)
    maes_lr.append(mae)
    rmses_lr.append(rmse)
    r2s_lr.append(r2)
    s += "\t(QR: %.2f %.2f %.2f)" % (mae,rmse,r2)
    corr, mae, mse, rae, rmse, r2 = compute_error(trues, preds_dl)
    maes_dl.append(mae)
    rmses_dl.append(rmse)
    r2s_dl.append(r2)
    s += "\t(DLQR: %.2f %.2f %.2f)  " % (mae,rmse,r2)
    corr, mae, mse, rae, rmse, r2 = compute_error(trues, preds_multi)
    maes_multi.append(mae)
    rmses_multi.append(rmse)
    r2s_multi.append(r2)
    s += "\t(MULTI: %.2f %.2f %.2f)" % (mae,rmse,r2)
    #print s

print "ERRORS:"
print "LR:    %.3f %.3f %.3f" % (np.mean(maes_lr),np.mean(rmses_lr),np.mean(r2s_lr)) 
print "DL:    %.3f %.3f %.3f" % (np.mean(maes_dl),np.mean(rmses_dl),np.mean(r2s_dl)) 
print "MULTI: %.3f %.3f %.3f" % (np.mean(maes_multi),np.mean(rmses_multi),np.mean(r2s_multi)) 
f_res.write("%.3f,%.3f,%.3f," % (np.mean(maes_lr),np.mean(maes_dl),np.mean(maes_multi)))
f_res.write("%.3f,%.3f,%.3f," % (np.mean(rmses_lr),np.mean(rmses_dl),np.mean(rmses_multi)))


# evaluate crossing quantiles

loss_qr = 0.0
loss_dl = 0.0
loss_multi = 0.0
for i in xrange(n_places):
     for k in xrange(len(quantiles)-1):
        q1 = predictions_q_lr[k][:,i]
        q2 = predictions_q_lr[k+1][:,i]
        loss_qr += np.sum(np.maximum(0.0, q1 - q2)) / len(q1)

        q1 = predictions_q[k][:,i,0,0] * removed_std_all[i][i_val:i_test] + removed_seasonality_all[i][i_val:i_test]
        q2 = predictions_q[k+1][:,i,0,0] * removed_std_all[i][i_val:i_test] + removed_seasonality_all[i][i_val:i_test]
        loss_dl += np.sum(np.maximum(0.0, q1 - q2)) / len(q1)

        q1 = predictions_multi[:,i,0,1+k] * removed_std_all[i][i_val:i_test] + removed_seasonality_all[i][i_val:i_test]
        q2 = predictions_multi[:,i,0,1+k+1] * removed_std_all[i][i_val:i_test] + removed_seasonality_all[i][i_val:i_test]
        loss_multi += np.sum(np.maximum(0.0, q1 - q2)) / len(q1)

print "CROSS LOSS QR:", loss_qr
print "CROSS LOSS DL:", loss_dl
print "CROSS LOSS MULTI:", loss_multi
f_res.write("%.3f,%.3f,%.3f," % (loss_qr,loss_dl,loss_multi))


sum_qr = 0
sum_dl = 0
sum_multi = 0
for i in xrange(n_places):
    for k in xrange(len(quantiles)-1):
        loss_qr = np.sum(predictions_q_lr[k][:,i] > predictions_q_lr[k+1][:,i])
        sum_qr += loss_qr
        loss_dl = np.sum(predictions_q[k][:,i,0,0] > predictions_q[k+1][:,i,0,0])
        sum_dl += loss_dl
        loss_multi = np.sum(predictions_multi[:,i,0,1+k] > predictions_multi[:,i,0,1+k+1])
        sum_multi += loss_multi

print "NUM CROSSES QR:", sum_qr
print "NUM CROSSES DL:", sum_dl
print "NUM CROSSES MULTI:", sum_multi
f_res.write("%d,%d,%d," % (sum_qr,sum_dl,sum_multi))


# evaluate tilted losses

def tilted_loss_np(q,y,f):
    e = (y-f)
    # The term inside k.mean is a one line simplification of the first equation
    return np.mean(q*e + np.clip(-e, K.epsilon(), np.inf), axis=-1)

sum_qr = 0
sum_dl = 0
sum_multi = 0
for i in xrange(n_places):
    for k in xrange(len(quantiles)):
        trues = y_test[:,0,i,0,0] * removed_std_all[i][i_val:i_test] + removed_seasonality_all[i][i_val:i_test]

        preds_qr = predictions_lr[0][:,i] 
        quant_qr = predictions_q_lr[k][:,i] 

        preds_dl = predictions[:,i,0,0] * removed_std_all[i][i_val:i_test] + removed_seasonality_all[i][i_val:i_test]
        quant_dl = predictions_q[k][:,i,0,0] * removed_std_all[i][i_val:i_test] + removed_seasonality_all[i][i_val:i_test]

        preds_multi = predictions_multi[:,i,0,0] * removed_std_all[i][i_val:i_test] + removed_seasonality_all[i][i_val:i_test]
        quant_multi = predictions_multi[:,i,0,k+1] * removed_std_all[i][i_val:i_test] + removed_seasonality_all[i][i_val:i_test]

        sum_qr += tilted_loss_np(quantiles[k],trues,quant_qr)
        sum_dl += tilted_loss_np(quantiles[k],trues,quant_dl)
        sum_multi += tilted_loss_np(quantiles[k],trues,quant_multi)

print "TILTED LOSS QR:", sum_qr
print "TILTED LOSS DL:", sum_dl
print "TILTED LOSS MULTI:", sum_multi
f_res.write("%.3f,%.3f,%.3f," % (sum_qr,sum_dl,sum_multi))


# evaluate quantiles

for pair in quantile_pairs:
    q_low, q_high = pair
    q_low_ix = quantiles.index(q_low)
    q_high_ix = quantiles.index(q_high)

    icps_qr = []
    mils_qr = []
    rmils_qr = []
    icps_dl = []
    mils_dl = []
    rmils_dl = []
    icps_multi = []
    mils_multi = []
    rmils_multi = []
    for i in xrange(n_places):
        trues = y_test[:,0,i,0,0] * removed_std_all[i][i_val:i_test] + removed_seasonality_all[i][i_val:i_test]

        preds_qr = predictions_lr[0][:,i] 
        lower_qr = predictions_q_lr[q_low_ix][:,i] 
        upper_qr = predictions_q_lr[q_high_ix][:,i] 

        preds_dl = predictions[:,i,0,0] * removed_std_all[i][i_val:i_test] + removed_seasonality_all[i][i_val:i_test]
        lower_dl = predictions_q[q_low_ix][:,i,0,0] * removed_std_all[i][i_val:i_test] + removed_seasonality_all[i][i_val:i_test]
        upper_dl = predictions_q[q_high_ix][:,i,0,0] * removed_std_all[i][i_val:i_test] + removed_seasonality_all[i][i_val:i_test]

        preds_multi = predictions_multi[:,i,0,0] * removed_std_all[i][i_val:i_test] + removed_seasonality_all[i][i_val:i_test]
        lower_multi = predictions_multi[:,i,0,q_low_ix+1] * removed_std_all[i][i_val:i_test] + removed_seasonality_all[i][i_val:i_test]
        upper_multi = predictions_multi[:,i,0,q_high_ix+1] * removed_std_all[i][i_val:i_test] + removed_seasonality_all[i][i_val:i_test]
        
        # evaluate quantiles
        s = "[%d]\tQuantiles:" % (i,)
        icp, mil, rmil, clc = eval_quantiles(lower_qr, upper_qr, trues, preds_qr)
        icps_qr.append(icp)
        mils_qr.append(mil)
        rmils_qr.append(rmil)
        s += "\t(QR: %.2f %.2f %.2f)" % (icp, mil, rmil)
        icp, mil, rmil, clc = eval_quantiles(lower_dl, upper_dl, trues, preds_dl)
        icps_dl.append(icp)
        mils_dl.append(mil)
        rmils_dl.append(rmil)
        s += "\t(DLQR: %.2f %.2f %.2f)  " % (icp, mil, rmil)
        icp, mil, rmil, clc = eval_quantiles(lower_multi, upper_multi, trues, preds_multi)
        icps_multi.append(icp)
        mils_multi.append(mil)
        rmils_multi.append(rmil)
        s += "\t(MULTI: %.2f %.2f %.2f)" % (icp, mil, rmil)

    print "QUANTILES (MEAN):"
    print "QR:    %.3f %.3f %.3f" % (np.mean(icps_qr),np.mean(mils_qr),np.mean(rmils_qr)) 
    print "DLQR:  %.3f %.3f %.3f" % (np.mean(icps_dl),np.mean(mils_dl),np.mean(rmils_dl)) 
    print "MULTI: %.3f %.3f %.3f" % (np.mean(icps_multi),np.mean(mils_multi),np.mean(rmils_multi)) 
    f_res.write("%.3f,%.3f,%.3f," % (np.mean(icps_qr),np.mean(icps_dl),np.mean(icps_multi)))
    f_res.write("%.3f,%.3f,%.3f," % (np.mean(mils_qr),np.mean(mils_dl),np.mean(mils_multi)))

    print "QUANTILES (MEDIAN):"
    print "QR:    %.3f %.3f %.3f" % (np.median(icps_qr),np.median(mils_qr),np.median(rmils_qr)) 
    print "DLQR:  %.3f %.3f %.3f" % (np.median(icps_dl),np.median(mils_dl),np.median(rmils_dl)) 
    print "MULTI: %.3f %.3f %.3f" % (np.median(icps_multi),np.median(mils_multi),np.median(rmils_multi)) 


f_res.close()


