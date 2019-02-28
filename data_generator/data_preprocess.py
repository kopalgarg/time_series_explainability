import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from datetime import timedelta, datetime
import pickle
from sklearn.impute import SimpleImputer


vital_IDs = ['HeartRate' , 'SysBP' , 'DiasBP' , 'MeanBP' , 'RespRate' , 'SpO2' , 'Glucose' ,'Temp']
lab_IDs = ['ANION GAP', 'ALBUMIN', 'BICARBONATE', 'BILIRUBIN', 'CREATININE', 'CHLORIDE', 'GLUCOSE', 'HEMATOCRIT', 'HEMOGLOBIN'
          'LACTATE', 'MAGNESIUM', 'PHOSPHATE', 'PLATELET', 'POTASSIUM', 'PTT', 'INR', 'PT', 'SODIUM', 'BUN', 'WBC']
eth_list = ['white', 'black', 'hispanic', 'asian', 'other']

eth_coder = lambda eth:0 if eth=='0' else eth_list.index(patient_data['ethnicity'].iloc[0])+1


def quantize_signal(signal, start, step_size, n_steps, value_column, charttime_column):
	quantized_signal = []
	quantized_counts = np.zeros((n_steps,))
	l = start
	u = start + timedelta(hours=step_size)
	for i in range(n_steps):
		signal_window = signal[value_column][(signal[charttime_column]>l) & (signal[charttime_column]<u)]
		quantized_signal.append(signal_window.mean())
		quantized_counts[i] = len(signal_window)
		l = u
		u = l + timedelta(hours=step_size)
	return quantized_signal,quantized_counts


def check_nan(A):
        A = np.array(A)
        nan_arr = np.isnan(A).astype(int)
        nan_count = np.count_nonzero(nan_arr)
        return nan_arr, nan_count


vital_data = pd.read_csv("./data/adult_icu_vital.gz", compression='gzip')#, nrows=1000) 
#print("Vitals:\n", vital_data[0:20])
vital_data = vital_data.dropna(subset=['vitalid'])

lab_data = pd.read_csv("./data/adult_icu_lab.gz", compression='gzip')#, nrows=1000) 
#print("Labs:\n", lab_data[0:20])
lab_data = lab_data.dropna(subset=['label'])


icu_id = vital_data.icustay_id.unique()
## features for every patient will be the list of vital IDs, gender(male=1, female=0), age, ethnicity(unknown=0 ,white=1, black=2, hispanic=3, asian=4, other=5), first_icu_stay(True=1, False=0)
x = np.zeros((len(icu_id), 12  , 48))
x_lab = np.zeros((len(icu_id), len(lab_IDs)  , 48))
x_impute = np.zeros((len(icu_id), 12  , 48))
y = np.zeros((len(icu_id),))
imp_mean = SimpleImputer(strategy="mean")

missing_ids = []
missing_map = np.zeros((len(icu_id), 12))
missing_map_lab = np.zeros((len(icu_id), len(lab_IDs)))

for i,id in enumerate(icu_id):
        patient_data = vital_data.loc[vital_data['icustay_id']==id]
        patient_data['vitalcharttime'] = patient_data['vitalcharttime'].astype('datetime64[s]')
        patient_lab_data = lab_data.loc[lab_data['icustay_id']==id]
        patient_lab_data['labcharttime'] = patient_lab_data['labcharttime'].astype('datetime64[s]')

        admit_time = patient_data['vitalcharttime'].min()
        #print('Patient %d admitted at '%(id),admit_time)
        n_missing_vitals = 0

	## Extract demographics and repeat them over time
        x[i,-4,:]= int(patient_data['gender'].iloc[0])
        x[i,-3,:]= int(patient_data['age'].iloc[0])
        x[i,-2,:]= eth_coder(patient_data['ethnicity'].iloc[0])
        x[i,-1,:]= int(patient_data['first_icu_stay'].iloc[0])
        y[i] = (int(patient_data['mort_icu'].iloc[0]))
                                    
	## Extract vital measurement informations
        vitals = patient_data.vitalid.unique()
        for vital in vitals:
                try:
                        vital_IDs.index(vital)
                        signal = patient_data[patient_data['vitalid']==vital]
                        quantized_signal, _ = quantize_signal(signal, start=admit_time, step_size=1, n_steps=48, value_column='vitalvalue', charttime_column='vitalcharttime')
                        nan_arr, nan_count = check_nan(quantized_signal)
                        x[i, vital_IDs.index(vital) , :] = np.array(quantized_signal)
                        if nan_count==48:
                                n_missing_vitals =+ 1
                                missing_map[i,vital_IDs.index(vital)]=1
                        else:
                                x_impute[i,:,:] = imp_mean.fit_transform(x[i,:,:].T).T
                except:
                        pass


	## Extract lab measurement informations
        labs = patient_lab_data.label.unique()
        for lab in labs:
                try:
                        lab_IDs.index(lab)
                        lab_measures = patient_lab_data[patient_lab_data['label']==lab]
                        quantized_lab , quantized_measures = quantize_signal(lab_measures, start=admit_time, step_size=1, n_steps=48, value_column='labvalue', charttime_column='labcharttime')
                        nan_arr, nan_count = check_nan(quantized_lab)
                        x_lab[i, lab_IDs.index(lab) , :] = np.array(quantized_measures)
                        if nan_count==48:
                                missing_map_lab[i,lab_IDs.index(lab)]=1
                except:
                        pass


        #if i==0:
        #        print(x_lab[i,:,:])
        ## Remove a patient that is missing a measurement for the entire 48 hours  
        if n_missing_vitals>0:
                missing_ids.append(i)


for i,vital in enumerate(vital_IDs):
        with open('./data/missing_stat.txt','a') as f:
                f.write("Missingness for %s: %.2f"%(vital,np.count_nonzero(missing_map[:,i])/len(icu_id)))
                f.write("\n")

for i,lab in enumerate(lab_IDs):
        with open('./data/missing_stat_lab.txt','a') as f:
                f.write("Missingness for %s: %.2f"%(lab,np.count_nonzero(missing_map_lab[:,i])/len(icu_id)))
                f.write("\n")


## Record statistics of the dataset, remove missing samples and save the signals
f = open("./data/stats.txt","a")
f.write('\n ******************* Before removing missing *********************')
f.write('\n Number of patients: '+ str(len(y))+'\n Number of patients who died within their stay: '+str(np.count_nonzero(y)))
x = np.delete(x, missing_ids, axis=0)
x_lab = np.delete(x_lab, missing_ids, axis=0)
x_impute = np.delete(x_impute, missing_ids, axis=0)
y = np.delete(y, missing_ids, axis=0)
all_data = np.concatenate((x_lab,x_impute),axis=1)

f.write('\n ******************* After removing missing *********************')
f.write('\n Final number of patients: '+str(len(y))+'\n Number of patients who died within their stay: '+str(np.count_nonzero(y)))
f.close()

samples = [ (all_data[i,:,:],y[i]) for i in range(len(y)) ]
with open('./data/patient_vital_preprocessed.pkl','wb') as f:
        pickle.dump(samples, f)





