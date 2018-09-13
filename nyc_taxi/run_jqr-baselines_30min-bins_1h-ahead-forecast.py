
import os.path
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
import statsmodels.formula.api as smf
from datetime import datetime
from sklearn import datasets, linear_model
from matplotlib import pyplot as plt
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

i_train = 2*24*90
i_val = i_train + 2*24*30
i_test = i_val + 2*24*60
num_train = 3000

quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.60, 0.70, 0.80, 0.90, 0.95]
quantile_pairs = [(0.05, 0.95), (0.10, 0.90), (0.20, 0.80), (0.30, 0.70), (0.40, 0.60)]

best_parameters = [[(0.1, 0.0001, 0.87370551789117612), (10.0, 0.01, 1.05583443831052), (10.0, 1.0, 2.0649885460315636), (1.0, 0.001, 2.1467231688980992), (1.0, 0.0001, 1.348960497043199), (0.1, 0.0001, 1.1937460917966876), (1.0, 0.001, 1.2315602924514693), (1.0, 0.001, 2.1723386735636012), (0.1, 0.0001, 6.8388192262540528), (1.0, 0.001, 1.06448317696585), (10.0, 0.1, 6.2521626368757302), (0.01, 0.0001, 2.1420965675824726)], [(10.0, 0.1, 0.35152620000317725), (0.0001, 0.0001, 1.6120343742583785), (10.0, 0.01, 1.6052653486865998), (0.0001, 0.0001, 2.8025496680371904), (10.0, 0.01, 4.7440015869013763), (10.0, 0.1, 5.6915306947717417), (0.0001, 0.0001, 2.823557196817279), (0.0001, 0.0001, 1.2414530702276494), (1.0, 0.001, 2.8798269038858066), (0.1, 0.001, 5.7240107286767303), (0.1, 0.01, 14.779979708103488), (1.0, 0.01, 7.1274663917530301)], [(10.0, 1.0, 0.16171940481584143), (0.1, 100.0, 0.0252397870725454), (1.0, 0.1, 0.07600896630377417), (0.0001, 0.0001, 1.186391013128566), (1.0, 0.001, 3.8110392684217018), (10.0, 0.1, 9.0238356430410374), (10.0, 10.0, 0.45818680667810896), (0.0001, 0.0001, 5.8456093583298756), (0.001, 0.001, 19.894358816731351), (1.0, 0.001, 14.33503450141121), (0.1, 0.0001, 14.33319320800776), (0.1, 0.0001, 3.3182583331355269)], [(10.0, 0.1, 0.29783703008041196), (100.0, 1.0, 0.31185411436598454), (10.0, 100.0, 0.18100546051434613), (10.0, 0.1, 0.65933438667192246), (1.0, 0.001, 3.2993722985574991), (10.0, 0.01, 1.9304824099295299), (0.1, 0.001, 1.5638656406730793), (1.0, 0.01, 14.023737395015964), (10.0, 0.1, 25.962014503478283), (0.01, 0.001, 23.813337535535577), (10.0, 0.1, 9.4204905731016648), (1.0, 100.0, 0.038781274483244955)], [(1.0, 0.001, 1.4225428219775904), (0.01, 0.0001, 4.1869799854005834), (0.001, 0.0001, 1.979092386814632), (0.001, 0.0001, 0.69759993021907263), (1.0, 0.001, 2.4499274768900716), (0.01, 0.0001, 1.6066545562419687), (1.0, 0.001, 4.8964341362019388), (0.0001, 0.0001, 19.992013534340607), (1.0, 0.01, 28.598335531643215), (1.0, 0.01, 20.585360111841602), (0.1, 0.0001, 15.409626490464408), (0.1, 100.0, 0.012121960546077891)], [(1.0, 0.001, 2.1819376535032649), (10.0, 0.01, 1.632523489350606), (0.001, 0.0001, 1.9843035234523541), (0.1, 0.001, 5.058494757742956), (10.0, 0.01, 2.9450016687985605), (0.0001, 0.001, 2.4433888339644674), (10.0, 0.01, 17.694086852868885), (0.001, 0.001, 28.278346715656863), (0.0001, 0.0001, 40.700406097213332), (10.0, 0.01, 14.10474190979556), (1.0, 0.001, 6.2107643508503374), (10.0, 100.0, 0.068385935584714702)], [(0.01, 0.0001, 1.7796226931048091), (10.0, 0.01, 4.6610450717910412), (1.0, 0.01, 3.8562597975173745), (100.0, 0.1, 2.0765155699703843), (10.0, 0.1, 2.9074318336927214), (0.1, 0.001, 12.976930129480106), (1.0, 0.01, 36.963740736964581), (1.0, 0.1, 34.980906251820628), (10.0, 0.1, 27.878187500111963), (0.001, 0.001, 13.627335001060427), (10.0, 10.0, 0.146692405967905), (10.0, 100.0, 0.16552711709783857)], [(1.0, 0.0001, 2.6211294496354056), (10.0, 0.01, 4.3894457496916672), (10.0, 10.0, 2.1174491493572516), (100.0, 0.1, 0.7533991184931772), (1.0, 0.01, 15.322991014972166), (0.1, 0.0001, 24.794868831944278), (0.001, 0.001, 14.678321252831974), (1.0, 0.01, 31.969087376629368), (10.0, 0.01, 13.378507277141091), (0.001, 0.001, 3.616222003283311), (1.0, 10.0, 0.024270613497058436), (10.0, 100.0, 0.16043998103125345)], [(10.0, 0.01, 2.4115508430606463), (10.0, 100.0, 0.80429083460631301), (1.0, 0.0001, 0.33003226709913225), (1.0, 0.001, 2.9546028687315165), (1.0, 0.01, 21.586488669138024), (0.0001, 0.0001, 22.484685534871286), (1.0, 0.01, 7.9456793511957997), (0.01, 0.001, 21.113742412768367), (10.0, 0.01, 4.8945076795361517), (1.0, 100.0, 0.047190969982313882), (10.0, 1.0, 0.12940368538674246), (1.0, 100.0, 0.034478817030737893)], [(0.01, 100.0, 0.0065630599655674233), (1.0, 0.1, 0.25055206885727577), (10.0, 0.1, 0.55384845440156194), (1.0, 0.001, 4.6707974256063789), (1.0, 0.001, 10.904928230808681), (1.0, 0.01, 4.6870944276278426), (1.0, 0.01, 15.884478249289566), (0.01, 0.001, 10.369996187935195), (0.01, 100.0, 0.021728799549225446), (10.0, 100.0, 0.063227838446310067), (10.0, 100.0, 0.070853203572223605), (0.01, 100.0, 0.04297585611490868)], [(10.0, 0.1, 0.2394885184858826), (1.0, 0.0001, 0.42550219436183001), (10.0, 0.01, 2.1961137249082379), (10.0, 0.01, 6.2938349106856952), (10.0, 0.01, 5.4075751138042074), (10.0, 0.1, 6.4707736087253327), (1.0, 0.01, 9.2473307890488528), (100.0, 1.0, 0.31610321612109749), (0.1, 100.0, 0.016224723845135808), (0.01, 100.0, 0.019593588542159663), (0.01, 100.0, 0.03340502969399943), (0.01, 10.0, 0.013843696368842594)], [(0.1, 0.001, 0.78811771226101657), (10.0, 0.1, 1.1089441418462314), (0.1, 0.0001, 1.4438853014973838), (10.0, 0.01, 5.9180006488024546), (10.0, 0.1, 6.7521211720669916), (0.0001, 0.0001, 7.9254311338780719), (10.0, 0.01, 2.7338508843942351), (0.01, 100.0, 0.0071034148222832803), (1.0, 100.0, 0.033450380205964789), (0.01, 100.0, 0.026170752174662769), (0.001, 100.0, 0.0095169218430362738), (0.0001, 1.0, 0.01092773044619463)]]


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


# ---------------- Linear Quantile Regression baselines

col_names = ["l%d" % (i,) for i in xrange(NUM_LAGS-STEPS_AHEAD)]
col_names.append("y")
qr_str = "y ~ " + ''.join(["+l%d" % (x,) for x in range(NUM_LAGS-STEPS_AHEAD)])[1:]

predictions_lr = []
for k in xrange(len(quantiles)):
    q = quantiles[k]
    print "\nTraining linear QR for quantile %.2f..." % (q,)
    
    preds_lr = np.zeros((i_test-i_val, nx, ny))
    for i in xrange(nx):
        for j in xrange(ny):
            regr = linear_model.LinearRegression()
            regr.fit(X_train[:,:,i,j], y_train[:,i,j])
            preds_lr[:,i,j] = regr.predict(X_test[:,:,i,j]) * matrix_std[i,j] + matrix_trend[i_val:i_test,i,j]

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
for i in xrange(nx):
    for j in xrange(ny):
        print "running JQR for cell:", i ,j

        #jqr_model = Quantile(probs=quantiles, kernel='DGauss', 
        #                    lbda=best_parameters[i][j][0], gamma=best_parameters[i][j][1], gamma_quantile=np.inf)
        jqr_model = Quantile(probs=quantiles, kernel='DGauss', 
                            lbda=best_parameters[i][j][0], gamma=best_parameters[i][j][1], gamma_quantile=1e-2)

        jqr_model.fit(X_train[rand_ix,:,i,j], y_train[rand_ix,i,j])

        preds = jqr_model.predict(X_test[:,:,i,j])

        trues = y_test[:,i,j] * matrix_std[i,j] + matrix_trend[i_val:i_test,i,j]
        preds_qr = predictions_lr[0][:,i,j] 

        corr, mae, mse, rae, rmse, r2 = compute_error(trues, preds_qr)
        sum_mae += mae
        sum_rmse += rmse
        sum_r2 += r2

        for k in xrange(len(quantile_pairs)):
            pair = quantile_pairs[k]
            q_low, q_high = pair
            q_low_ix = quantiles.index(q_low)
            q_high_ix = quantiles.index(q_high)

            lower_qr = preds[q_low_ix,:] * matrix_std[i,j] + matrix_trend[i_val:i_test,i,j]
            upper_qr = preds[q_high_ix,:] * matrix_std[i,j] + matrix_trend[i_val:i_test,i,j]

            icp, mil, rmil, clc = eval_quantiles(lower_qr, upper_qr, trues, preds_qr)
            print pair, "\t(JQR: %.3f %.3f %.3f)" % (icp, mil, rmil)
            sum_icp[k] += icp
            sum_mil[k] += mil
            sum_rmil[k] += rmil

        # tilted loss
        tloss = 0.0
        for k in xrange(len(quantiles)):
            quant_qr = preds[k,:] * matrix_std[i,j] + matrix_trend[i_val:i_test,i,j]
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

n_places = nx*ny
print n_places
f_res.write("%.3f,%3f," % (sum_mae/n_places, sum_rmse/n_places))
f_res.write("%.3f,%d,%.3f" % (sum_closs, total_crosses, sum_tloss))
for j in xrange(len(sum_icp)):
    f_res.write(",%.3f,%.3f" % (sum_icp[j]/n_places, sum_mil[j]/n_places))

f_res.close()



