
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import os.path
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
import statsmodels.formula.api as smf
from datetime import datetime
from sklearn import datasets, linear_model
from matplotlib import pyplot as plt
from matplotlib import cm
from keras.layers.core import Dense, Activation, Dropout
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
N_EPOCHS_MULTI = 150 

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
    model.add(Dropout(0.75))

    model.add(Dense(units=num_outputs, 
                    kernel_regularizer=regularizers.l2(0.0001),
             ))
    model.add(Activation("linear"))

    model.compile(loss=loss, optimizer='nadam')
    
    return model


print "\nTraining NN for mean..."

# checkpoint best model
checkpoint = ModelCheckpoint("weights_mean.best.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# fit model to the mean
model_mean = build_model(loss="mse", num_outputs=1)
model_mean.fit(
    X_train[:,:,:,:,np.newaxis],
    y_train[:,:,:,np.newaxis],
    batch_size=128,
    epochs=N_EPOCHS, #100
    validation_data=(X_val[:,:,:,:,np.newaxis],y_val[:,:,:,np.newaxis]),
    callbacks=[checkpoint],
    verbose=0)   


# load weights
model_mean.load_weights("weights_mean.best.hdf5")

# make predictions
predictions = model_mean.predict(X_test[:,:,:,:,np.newaxis])


models_q = []
for q in quantiles:
    print "\nTraining NN for quantile %.2f..." % (q,)
    
    # checkpoint best model
    checkpoint = ModelCheckpoint("weights_q%2f.best.hdf5" % (q,), monitor='val_loss', verbose=0, save_best_only=True, mode='min')

    # fit model to the 5% quantile
    model_q = build_model(loss=lambda y,f: tilted_loss(q,y,f))
    model_q.fit(
        X_train[:,:,:,:,np.newaxis],
        y_train[:,:,:,np.newaxis],
        batch_size=128,
        epochs=N_EPOCHS, 
        validation_data=(X_val[:,:,:,:,np.newaxis],y_val[:,:,:,np.newaxis]),
        callbacks=[checkpoint],
        verbose=0)  
    models_q.append(model_q)

predictions_q = []
for i in xrange(len(quantiles)):
    q = quantiles[i]
    print "\nMaking predictions for quantile %.2f..." % (q,)
    
    # load weights
    models_q[i].load_weights("weights_q%2f.best.hdf5" % (q,))

    # make predictions
    predictions_q.append(models_q[i].predict(X_test[:,:,:,:,np.newaxis]))


# ---------------- Linear Quantile Regression baselines

col_names = ["l%d" % (i,) for i in xrange(NUM_LAGS-STEPS_AHEAD)]
col_names.append("y")
qr_str = "y ~ " + ''.join(["+l%d" % (x,) for x in range(NUM_LAGS-STEPS_AHEAD)])[1:]

predictions_lr = []
predictions_q_lr = []
for k in xrange(len(quantiles)):
    q = quantiles[k]
    print "\nTraining linear QR for quantile %.2f..." % (q,)
    
    preds_lr = np.zeros((i_test-i_val, nx, ny))
    preds_qr = np.zeros((i_test-i_val, nx, ny))
    for i in xrange(nx):
        for j in xrange(ny):
            regr = linear_model.LinearRegression()
            regr.fit(X_train[:,:,i,j], y_train[:,i,j])
            preds_lr[:,i,j] = regr.predict(X_test[:,:,i,j]) * matrix_std[i,j] + matrix_trend[i_val:i_test,i,j]
            
            data = pd.DataFrame(data = np.hstack([X_train[:,:,i,j], y_train[:,i,j,np.newaxis]]), columns = col_names)
            data_test = pd.DataFrame(data = np.hstack([X_test[:,:,i,j], y_test[:,i,j,np.newaxis]]), columns = col_names)

            qr = smf.quantreg(qr_str, data)
            res = qr.fit(q=q)
            preds_qr[:,i,j] = res.predict(data_test) * matrix_std[i,j] + matrix_trend[i_val:i_test,i,j]
    
    predictions_lr.append(preds_lr)
    predictions_q_lr.append(preds_qr)


# ---------------- Simultaneous Mean and Quantiles NN

def multi_tilted_loss(quantiles,y,f):
    # The term inside k.mean is a one line simplification of the first equation
    loss = K.mean(K.square(y[:,:,:,0]-f[:,:,:,0]), axis=-1)
    for k in xrange(len(quantiles)):
        q = quantiles[k]
        e = (y[:,:,:,k+1]-f[:,:,:,k+1])
        loss += K.mean(q*e + K.clip(-e, K.epsilon(), np.inf), axis=-1)
    return loss


def multi_tilted_loss_cross(quantiles,y,f):
    print y
    print f
    # The term inside k.mean is a one line simplification of the first equation
    loss = K.mean(K.square(y[:,:,:,0]-f[:,:,:,0]), axis=-1)
    for k in xrange(len(quantiles)):
        q = quantiles[k]
        e = (y[:,:,:,k+1]-f[:,:,:,k+1])
        loss += K.mean(q*e + K.clip(-e, K.epsilon(), np.inf), axis=-1)
        
    for k in xrange(len(quantiles)-1):
        cross_loss = K.mean(K.maximum(.0,f[:,:,:,k+1]-f[:,:,:,k+2]), axis=-1)
        
    loss += cross_loss
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
        
    for k in xrange(len(quantiles)-1):
        cross_loss = K.mean(K.maximum(.0,f[:,:,:,k+1]-f[:,:,:,k+2]), axis=-1)
        
    loss += cross_loss
    return loss


y_traink = y_train[:,:,:,np.newaxis]
for k in xrange(len(quantiles)):
    y_traink = np.concatenate((y_traink, y_train[:,:,:,np.newaxis]), axis=3)

y_valk = y_val[:,:,:,np.newaxis]
for k in xrange(len(quantiles)):
    y_valk = np.concatenate((y_valk, y_val[:,:,:,np.newaxis]), axis=3)

print "\nTraining multi-quantile NN..."

# checkpoint best model
checkpoint = ModelCheckpoint("weights_multi.best.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# fit model to the 95% quantile
model_multi = build_model(loss=lambda y,f: multi_tilted_loss_weighted(quantiles,y,f), num_outputs=1+len(quantiles))
model_multi.fit(
    X_train[:,:,:,:,np.newaxis],
    y_traink,
    batch_size=128,
    epochs=N_EPOCHS_MULTI, # 150
    validation_data=(X_val[:,:,:,:,np.newaxis],y_valk),
    callbacks=[checkpoint],
    verbose=2)

# load weights
model_multi.load_weights("weights_multi.best.hdf5")

# make predictions
predictions_multi = model_multi.predict(X_test[:,:,:,:,np.newaxis])


# ---------------- Evaluate different methods

print "\nEvaluating different methods..."

fname = "results_crossloss_%dlags" % (NUM_LAGS,)
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
for i in xrange(nx):
    for j in xrange(ny):
        trues = y_test[:,i,j] * matrix_std[i,j] + matrix_trend[i_val:i_test,i,j]
        preds_qr = predictions_lr[0][:,i,j] 
        preds_dl = predictions[:,i,j,0] * matrix_std[i,j] + matrix_trend[i_val:i_test,i,j]
        preds_multi = predictions_multi[:,i,j,0] * matrix_std[i,j] + matrix_trend[i_val:i_test,i,j]

        # evaluate prediction means
        s = "[%d,%d]\tMean:" % (i,j)
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
for i in xrange(12):
    for j in xrange(12):
         for k in xrange(len(quantiles)-1):
            q1 = predictions_q_lr[k][:,i,j]
            q2 = predictions_q_lr[k+1][:,i,j]
            loss_qr += np.sum(np.maximum(0.0, q1 - q2)) / len(q1)
            
            q1 = predictions_q[k][:,i,j,0] * matrix_std[i,j] + matrix_trend[i_val:i_test,i,j]
            q2 = predictions_q[k+1][:,i,j,0] * matrix_std[i,j] + matrix_trend[i_val:i_test,i,j]
            loss_dl += np.sum(np.maximum(0.0, q1 - q2)) / len(q1)
            
            q1 = predictions_multi[:,i,j,1+k] * matrix_std[i,j] + matrix_trend[i_val:i_test,i,j]
            q2 = predictions_multi[:,i,j,1+k+1] * matrix_std[i,j] + matrix_trend[i_val:i_test,i,j]
            loss_multi += np.sum(np.maximum(0.0, q1 - q2)) / len(q1)

print "CROSS LOSS QR:", loss_qr
print "CROSS LOSS DL:", loss_dl
print "CROSS LOSS MULTI:", loss_multi
f_res.write("%.3f,%.3f,%.3f," % (loss_qr,loss_dl,loss_multi))


sum_qr = 0
sum_dl = 0
sum_multi = 0
for i in xrange(12):
    for j in xrange(12):
        for k in xrange(len(quantiles)-1):
            loss_qr = np.sum(predictions_q_lr[k][:,i,j] > predictions_q_lr[k+1][:,i,j])
            sum_qr += loss_qr
            loss_dl = np.sum(predictions_q[k][:,i,j,0] > predictions_q[k+1][:,i,j,0])
            sum_dl += loss_dl
            loss_multi = np.sum(predictions_multi[:,i,j,1+k] > predictions_multi[:,i,j,1+k+1])
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
for i in xrange(12):
    for j in xrange(12):
        for k in xrange(len(quantiles)):
            trues = y_test[:,i,j] * matrix_std[i,j] + matrix_trend[i_val:i_test,i,j]

            preds_qr = predictions_lr[0][:,i,j] 
            quant_qr = predictions_q_lr[k][:,i,j] 

            preds_dl = predictions[:,i,j,0] * matrix_std[i,j] + matrix_trend[i_val:i_test,i,j]
            quant_dl = predictions_q[k][:,i,j,0] * matrix_std[i,j] + matrix_trend[i_val:i_test,i,j]

            preds_multi = predictions_multi[:,i,j,0] * matrix_std[i,j] + matrix_trend[i_val:i_test,i,j]
            quant_multi = predictions_multi[:,i,j,k+1] * matrix_std[i,j] + matrix_trend[i_val:i_test,i,j]

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
    for i in xrange(nx):
        for j in xrange(ny):
            trues = y_test[:,i,j] * matrix_std[i,j] + matrix_trend[i_val:i_test,i,j]

            preds_qr = predictions_lr[0][:,i,j] 
            lower_qr = predictions_q_lr[q_low_ix][:,i,j] 
            upper_qr = predictions_q_lr[q_high_ix][:,i,j] 

            preds_dl = predictions[:,i,j,0] * matrix_std[i,j] + matrix_trend[i_val:i_test,i,j]
            lower_dl = predictions_q[q_low_ix][:,i,j,0] * matrix_std[i,j] + matrix_trend[i_val:i_test,i,j]
            upper_dl = predictions_q[q_high_ix][:,i,j,0] * matrix_std[i,j] + matrix_trend[i_val:i_test,i,j]

            preds_multi = predictions_multi[:,i,j,0] * matrix_std[i,j] + matrix_trend[i_val:i_test,i,j]
            lower_multi = predictions_multi[:,i,j,q_low_ix+1] * matrix_std[i,j] + matrix_trend[i_val:i_test,i,j]
            upper_multi = predictions_multi[:,i,j,q_high_ix+1] * matrix_std[i,j] + matrix_trend[i_val:i_test,i,j]
            
            # evaluate quantiles
            s = "[%d,%d]\tQuantiles:" % (i,j)
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
            #print s

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

