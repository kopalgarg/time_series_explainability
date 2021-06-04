import timesynth as ts
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from scipy.signal import butter, lfilter, freqz
import pickle as pkl
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Filter requirements.
order = 6
fs = 30.0       # sample rate, Hz
cutoff = 6.8 # desired cutoff frequency of the filter, Hz

# Get the filter coefficients so we can check its frequency response.
b, a = butter_lowpass(cutoff, fs, order)

w, h = freqz(b, a, worN=8000)
plt.subplot(1, 1, 1)
plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
plt.axvline(cutoff, color='k')
plt.xlim(0, 0.5*fs)
plt.title("Lowpass Filter Frequency Response")
plt.xlabel('Frequency [Hz]')
plt.grid()


def main(plot,n_samples,  Tt=100):
    signal_in = []
    thresholds = []
    trend_style=[]
    for i in range(n_samples):
        x,t,trend = generate_sample(plot, Tt=Tt)
        signal_in.append(x)
        thresholds.append(t)
        trend_style.append(trend)

    print('samples done!')
    signal_in = np.array(signal_in)

    n_train = int(0.8*n_samples)
    x_train = signal_in[0:n_train,:,:]
    thresholds_train = thresholds[0:n_train]
    n_test = n_samples - n_train

    x_test = signal_in[n_train:,:,:]
    thresholds_test = thresholds[n_train:]

    if 0:
        scaler = MinMaxScaler()
        x_train_flat = scaler.fit_transform(np.reshape(x_train,[x_train.shape[0],-1]))
        x_train_n = np.reshape(x_train_flat,x_train.shape)
        x_test_flat = scaler.transform(np.reshape(x_test,[x_test.shape[0],-1]))
        x_test_n = np.reshape(x_test_flat,x_test.shape)
    else:
        x_train_n = x_train
        x_test_n = x_test
        
    if 1:
        x1_train_lpf = np.array([butter_lowpass_filter(x[0,:], cutoff, fs, order) for x in x_train_n])
        x2_train_lpf = np.array([butter_lowpass_filter(x[1,:], cutoff, fs, order) for x in x_train_n])
        x3_train_lpf = np.array([butter_lowpass_filter(x[2,:], cutoff, fs, order) for x in x_train_n])
        x_train_n = np.stack([x1_train_lpf, x2_train_lpf, x3_train_lpf],axis=1)

        x1_test_lpf = np.array([butter_lowpass_filter(x[0,:], cutoff, fs, order) for x in x_test_n])
        x2_test_lpf = np.array([butter_lowpass_filter(x[1,:], cutoff, fs, order) for x in x_test_n])
        x3_test_lpf = np.array([butter_lowpass_filter(x[2,:], cutoff, fs, order) for x in x_test_n])
        x_test_n = np.stack([x1_test_lpf, x2_test_lpf, x3_test_lpf],axis=1)

    kappa=1.5
    y_train_f0 = []
    y_train_f1 = []
    y_train_f2 = []

    y_test_f0 = []
    y_test_f1 = []
    y_test_f2 = []

    ground_truth_importance_train_f0=[]
    ground_truth_importance_train_f1=[]
    ground_truth_importance_train_f2=[]

    ground_truth_importance_test_f0=[]
    ground_truth_importance_test_f1=[]
    ground_truth_importance_test_f2=[]


    pvec=[0.7,0.3]
    lamda_poisson = 3
    
    for i in range(n_train):

        # f0
        gt_t_f0 = np.zeros((Tt))
        add_spikes = np.random.choice([0,1],1,p=pvec)
        if add_spikes:
            nts = np.random.poisson(lamda_poisson,1)
            
        else:
            nts=0
        t_vec_f0 = np.sort(np.random.choice(list(range(Tt)),nts,replace=False))
        
        x_train_n[i,0,t_vec_f0] +=   kappa
        
        if len(t_vec_f0)>0:
            gt_t_f0[t_vec_f0] = 1
            
        y_vec_f0 = np.zeros((Tt))
        if len(t_vec_f0)>0:
            y_vec_f0[t_vec_f0] = 1
        y_train_f0.append(y_vec_f0)
        ground_truth_importance_train_f0.append(gt_t_f0)

        # f1
        gt_t_f1 = np.zeros((Tt))
        add_spikes = np.random.choice([0,1],1,p=pvec)
        if add_spikes:
            nts = np.random.poisson(lamda_poisson,1)
        else:
            nts=0
        t_vec_f1 = np.sort(np.random.choice(list(range(Tt)),nts,replace=False))
        x_train_n[i,1,t_vec_f1] +=   kappa
        if len(t_vec_f1)>0:
            gt_t_f1[t_vec_f1] = 1
            
        y_vec_f1 = np.zeros((Tt))
        if len(t_vec_f1)>0:
            y_vec_f1[t_vec_f1] = 1

        y_train_f1.append(y_vec_f1)
        ground_truth_importance_train_f1.append(gt_t_f1)

        # f2
        gt_t_f2 = np.zeros((Tt))
        add_spikes = np.random.choice([0,1],1,p=pvec)
        if add_spikes:
            nts = np.random.poisson(lamda_poisson,1)
        else:
            nts=0
        t_vec_f2 = np.sort(np.random.choice(list(range(Tt)),nts,replace=False))
        x_train_n[i,2,t_vec_f2] +=   kappa
        if len(t_vec_f2)>0:
            gt_t_f2[t_vec_f2] = 1
            
        y_vec_f2 = np.zeros((Tt))
        if len(t_vec_f2)>0:
            y_vec_f2[t_vec_f2] = 1
        y_train_f2.append(y_vec_f2)
        ground_truth_importance_train_f2.append(gt_t_f2)


    for i in range(x_test_n.shape[0]):
        # f0
        gt_t_f0 = np.zeros((Tt))
        add_spikes = np.random.choice([0,1],1,p=pvec)
        if add_spikes:
            nts = np.random.poisson(lamda_poisson,1)
            
        else:
            nts=0
        t_vec_f0 = np.sort(np.random.choice(list(range(Tt)),nts,replace=False))
        
        x_test_n[i,0,t_vec_f0] +=   kappa
        
        if len(t_vec_f0)>0:
            gt_t_f0[t_vec_f0] = 1
            
        y_vec_f0 = np.zeros((Tt))
        if len(t_vec_f0)>0:
            y_vec_f0[t_vec_f0] = 1
        y_test_f0.append(y_vec_f0)
        ground_truth_importance_test_f0.append(gt_t_f0)

        # f1
        gt_t_f1 = np.zeros((Tt))
        add_spikes = np.random.choice([0,1],1,p=pvec)
        if add_spikes:
            nts = np.random.poisson(lamda_poisson,1)
        else:
            nts=0
        t_vec_f1 = np.sort(np.random.choice(list(range(Tt)),nts,replace=False))
        x_test_n[i,1,t_vec_f1] +=   kappa
        if len(t_vec_f1)>0:
            gt_t_f1[t_vec_f1] = 1
            
        y_vec_f1 = np.zeros((Tt))
        if len(t_vec_f1)>0:
            y_vec_f1[t_vec_f1] = 1

        y_test_f1.append(y_vec_f1)
        ground_truth_importance_test_f1.append(gt_t_f1)

        # f2
        gt_t_f2 = np.zeros((Tt))
        add_spikes = np.random.choice([0,1],1,p=pvec)
        if add_spikes:
            nts = np.random.poisson(lamda_poisson,1)
        else:
            nts=0
        t_vec_f2 = np.sort(np.random.choice(list(range(Tt)),nts,replace=False))
        x_test_n[i,2,t_vec_f2] +=   kappa
        if len(t_vec_f2)>0:
            gt_t_f2[t_vec_f2] = 1
            
        y_vec_f2 = np.zeros((Tt))
        if len(t_vec_f2)>0:
            y_vec_f2[t_vec_f2] = 1
        y_test_f2.append(y_vec_f2)
        ground_truth_importance_test_f2.append(gt_t_f2)



    y_train_f0 = np.array(y_train_f0)
    y_train_f1 = np.array(y_train_f1)
    y_train_f2 = np.array(y_train_f2)

    y_test_f0 = np.array(y_test_f0)
    y_test_f1 = np.array(y_test_f1)
    y_test_f2 = np.array(y_test_f2)

    ground_truth_imp_train_mat = np.zeros((n_train,3,Tt))
    ground_truth_imp_test_mat = np.zeros((n_test,3,Tt))

    ground_truth_imp_train_mat[:,0,:] = ground_truth_importance_train_f0
    ground_truth_imp_train_mat[:,1,:] = ground_truth_importance_train_f1
    ground_truth_imp_train_mat[:,2,:] = ground_truth_importance_train_f2

    ground_truth_imp_test_mat[:,0,:] = ground_truth_importance_test_f0
    ground_truth_imp_test_mat[:,1,:] = ground_truth_importance_test_f1
    ground_truth_imp_test_mat[:,2,:] = ground_truth_importance_test_f2

    y_train = y_train_f0 + y_train_f1 + y_train_f2
    y_train = np.where(y_train>1, 1, y_train) 

    y_test = y_test_f0 + y_test_f1 + y_test_f2
    y_test = np.where(y_test>1, 1, y_test) 

 
    if plot:
        for i in range(x_train_n.shape[0]):
            plt.plot(x_train_n[i,0,:], label='x1')
            plt.plot(x_train_n[i,1,:], label='x2')
            plt.plot(x_train_n[i,2,:], label='x3')
            plt.plot(y_train[i,:])
            #plt.ylim([-5,5])
            plt.title('Sample style: %s'%(trend_style[i]))

            if isinstance(thresholds[i],np.ndarray):
                for thresh in thresholds[i]:
                    plt.axvline(x=thresh, color='grey')
            else:
                plt.axvline(x=thresholds[i], color='grey')

            plt.legend()
            plt.show()
    return x_train_n[:,:,:],y_train,x_test_n[:,:,:],y_test,thresholds_train,thresholds_test, ground_truth_imp_train_mat[:,:,:], ground_truth_imp_test_mat[:,:,:]

def generate_sample(plot, Tt=100):
    trend_style='hill'
    noise = ts.noise.GaussianNoise(std=0.001)
    x1 = ts.signals.NARMA(order=2,coefficients=[.5,.5,1.5,.5],seed=random.seed())
    x1_ts = ts.TimeSeries(x1, noise_generator=noise)
    x1_sample, signals, errors = x1_ts.sample(np.array(range(Tt)))
    x1_sample += 0.003*np.array(range(Tt)) 
    x1_sample = x1_sample/(max(x1_sample)-min(x1_sample)) # - 1
    
    noise = ts.noise.GaussianNoise(std=0.001)
    x2 = ts.signals.NARMA(order=2,coefficients=[.5,.5,1.5,.5],seed=random.seed())
    x2_ts = ts.TimeSeries(x2,noise_generator=noise)
    x2_sample,signals,errors = x2_ts.sample(np.array(range(Tt)))
    x2_sample += 0.065*np.array(range(Tt)) 
    x2_sample /=(max(x2_sample)-min(x2_sample))
    
    noise = ts.noise.GaussianNoise(std=0.001)
    x3 = ts.signals.NARMA(order=2,seed=random.seed())
    x3_ts = ts.TimeSeries(x3, noise_generator=noise)
    x3_sample, signals, errors = x3_ts.sample(np.array(range(Tt)))
    x3_sample = x3_sample/(max(x3_sample)-min(x3_sample)) - 0.5
    
    t = np.array(np.zeros(4))
    
    return np.stack([x1_sample, x2_sample, x3_sample]), t, trend_style


def save_data(path,array):
    with open(path,'wb') as f:
        pkl.dump(array, f)


def logistic(x):
    return 1./(1+np.exp(-1*x))


if __name__=='__main__':
    if not os.path.exists('./data'):
        os.mkdir('./data')
    n_samples = 100 #3000
    x_train_n,y_train,x_test_n,y_test,thresholds_train,thresholds_test, gt_importance_train, gt_importance_test = main(n_samples=n_samples, plot=False)
    if not os.path.exists('./data/simulated_spike_data'):
        os.mkdir('./data/simulated_spike_data')
    save_data('./data/simulated_spike_data/x_train.pkl', x_train_n)
    save_data('./data/simulated_spike_data/y_train.pkl', y_train)
    save_data('./data/simulated_spike_data/x_test.pkl', x_test_n)
    save_data('./data/simulated_spike_data/y_test.pkl', y_test)
    save_data('./data/simulated_spike_data/thresholds_train.pkl', thresholds_train)
    save_data('./data/simulated_spike_data/thresholds_test.pkl', thresholds_test)
    save_data('./data/simulated_spike_data/gt_train.pkl', gt_importance_train)
    save_data('./data/simulated_spike_data/gt_test.pkl', gt_importance_test)
    print(gt_importance_train.shape)
