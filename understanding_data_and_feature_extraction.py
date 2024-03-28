## Imports

import pandas as pd
import numpy as np
import heartpy as hp
import pickle
import matplotlib.pyplot as plt
import warnings
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")

## Load .pkl from input_data (Compatible with Pandas v 1.5.3)

with open('input_data/ppg_one_minute_8.pkl', 'rb') as f:
    data = pickle.load(f)

print(data.head(5))
print(f'(Rows, Columns) = {data.shape}')
print(f'1 min PPG measurements = 125Hz * 60 = {len(data.PPG[0])}')

## Understand IDs in file
# Conclusion: Each ID resembles one patient. Each file can contain multiple patients.

counter = 0
patient_id = "30/3003173/"
for index, item in data.iterrows():
    if item['ID'] == patient_id: counter += 1

print(f'Number of patients in file: {len(data.ID.unique())}')
print(f'Patient with ID {patient_id} has {counter} 1 min PPG measurements')

## Show indices where the selected PPG signal has a nan value
# Conclusion: Data is not cleaned properly. Nan values are usually in a continuous order
# Possible solution - interpolation when null values are less than 5%

position = 0
for i in data.PPG[0]:
    if np.isnan(i):
        if not np.isnan(data.PPG[0][position - 1]) and position != 0:
            print(position-1, data.PPG[0][position - 1])
        print(position, i)
        if not np.isnan(data.PPG[0][position + 1]) and position != len(data.PPG[0]):
            print(position+1, data.PPG[0][position + 1])
    position += 1

## Visualisation before interpolation - shows a segment of the PPG signal fluctuations including the impact of the Nan values

plt.figure(figsize=(12,4))
plt.plot(data.PPG[0][:200])
plt.show()

## File summary and interpolation

signals_containing_nan = []
signals_containing_nan_5pc = []
low_spo2_below_95 = 0  # number of entries where spo2 < 95 and at least 1 Nan value for PPG

data['interpolation_flag'] = 0
for index, item in data.iterrows():
    ppg_signal = item['PPG']
    spo2 = item['SpO2']
    if np.isnan(ppg_signal).any():
        signals_containing_nan.append((index, spo2))
        if np.isnan(ppg_signal).mean() <= 0.05:
            signals_containing_nan_5pc.append((index, spo2))
        if spo2 < 95:
            low_spo2_below_95 += 1

    if np.isnan(ppg_signal).any():
        if np.isnan(ppg_signal).mean() <= 0.05:
            # Generate an array of indices without NaN values
            valid_indices = np.arange(len(ppg_signal))[~np.isnan(ppg_signal)]
            # Interpolate NaN values in array 'ppg_signal'
            interpolated_values = np.interp(np.arange(len(ppg_signal)), valid_indices, ppg_signal[valid_indices])
            # Set interpolated values to the current PPG array
            data.at[index, 'PPG'] = interpolated_values
            data.at[index, 'interpolation_flag'] = 1
        else:
            # Drop the row if the mean of NaN values is greater than 0.05
            data = data.drop(index)

print(f'# of unique patients inside one file: {data["ID"].nunique()}')
print(f'Total number of PPG signals: {data.shape[0]}')
print(f'# of PPG signals with at least one Nan: {len(signals_containing_nan)}')
print(f'# of PPG signals with a maximum of 5% Nan: {len(signals_containing_nan_5pc)}')
print(f'# of PPG signals with more than 5% Nan: {len(signals_containing_nan) - len(signals_containing_nan_5pc)}')

if len(signals_containing_nan) != 0:
  print(f'% of PPG signals with at least one Nan [PPG-Including-Nan]: {(len(signals_containing_nan)/data.shape[0])*100:.3f}%')
  print(f'% of PPG signals with a maximum of 5% Nan, from PPG-Including-Nan: {(len(signals_containing_nan_5pc)/len(signals_containing_nan))*100:.3f}%')
  print(f'% of PPG signals with more than 5% Nan, from PPG-Including-Nan: {(1-len(signals_containing_nan_5pc)/len(signals_containing_nan))*100:.3f}%')
  print(f'% of entries where SpO2 is lower than 95, from PPG-Including-Nan: {low_spo2_below_95/(len(signals_containing_nan))*100:.3f}%')

## Visualisation after interpolation - shows a segment of the PPG signal fluctuations including the impact of the interpolated Nan values
# Conclusion - The visualisation asserts that the use of interpolation is a viable solution to filling missing data

plt.figure(figsize=(12,4))
plt.plot(data.PPG[0][:200])
plt.show()

## Return selected rows for patient

# Algorithm for selecting rows:
# 1. Where SpO2 < 95
# 2. Equal nb. of rows where SpO2 >= 95. If not possible, take all rows where SpO2 >= 95.

def get_measurements(group):
    data = pd.DataFrame(columns=group.columns)
    for index, item in group.iterrows():
        item['PPG'] = hp.filter_signal(item['PPG'], cutoff = [0.5, 3.75], sample_rate = 125, order = 2, filtertype='bandpass')
        spo2 = item['SpO2']
        if spo2 < 95:
            data = data.append(item, ignore_index=True)
            group.drop(index, inplace=True)

    group_size = group.shape[0]
    data_size = data.shape[0]

    if data_size < group_size:  # more data with SpO2 >= 95
        num_rows_to_be_selected = data_size
    else:                       # more data with SpO2 < 95
        num_rows_to_be_selected = group_size

    random_rows = group.sample(n=num_rows_to_be_selected, replace=False)
    data = data.append(random_rows, ignore_index=True)

    return data

## Extracting Respiratory features

df = pd.DataFrame()
data_groupedBy_patientID = data.groupby('ID')

for _, group in data_groupedBy_patientID:

    data_current_patient = get_measurements(group)

    for index, item in data_current_patient.iterrows():
        ppg_signal = item['PPG']

        # Amplitude Features
        mean_amplitude = np.mean(ppg_signal)
        max_amplitude = np.max(ppg_signal)
        min_amplitude = np.min(ppg_signal)
        std_amplitude = np.std(ppg_signal)

        # Frequency Domain Features
        # fft = np.fft.fft(respiratory_signal)
        # power_spectrum = np.abs(fft)**2
        # dominant_frequency = np.argmax(power_spectrum)

        # Time-Domain Features
        peaks, _ = find_peaks(ppg_signal, distance=1)
        breath_duration = len(peaks)
        respiratory_rate = len(peaks) / len(ppg_signal) * 60

        # Variability Features
        rrv = np.diff(peaks)
        rrv_mean = np.mean(rrv)
        rrv_std = np.std(rrv)

        # Shape Features
        rise_time = np.argmax(ppg_signal) - np.argmin(ppg_signal)
        fall_time = np.argmax(ppg_signal[::-1]) - np.argmin(ppg_signal[::-1])

        respiratory_features = {
            'mean_amplitude': mean_amplitude,
            'max_amplitude': max_amplitude,
            'min_amplitude': min_amplitude,
            'std_amplitude': std_amplitude,
            'breath_duration': breath_duration,
            'respiratory_rate': respiratory_rate,
            'rrv_mean': rrv_mean,
            'rrv_std': rrv_std,
            'rise_time': rise_time,
            'fall_time': fall_time,
        }

        for feature_name, feature_value in respiratory_features.items():
            data_current_patient.at[index, feature_name] = feature_value

    df = df.append(data_current_patient)

df.to_csv('output_data/respiratory-processed_ppg_one_minute_8.csv', index=False)

## Extracting HeartPy features

df = pd.DataFrame()
data_groupedBy_patientID = data.groupby('ID')

for _, group in data_groupedBy_patientID:

    data_current_patient = get_measurements(group)

    for index, item in data_current_patient.iterrows():
        ppg_signal = item['PPG']

        try:
            _, measures = hp.process(np.array(ppg_signal), sample_rate=125)

            hrv_features = {
                'bpm': measures['bpm'],
                'ibi': measures['ibi'],
                'sdnn': measures['sdnn'],
                'sdsd': measures['sdsd'],
                'rmssd': measures['rmssd'],
                'pnn20': measures['pnn20'],
                'pnn50': measures['pnn50'],
                'hr_mad': measures['hr_mad'],
                'sd1': measures['sd1'],
                'sd2': measures['sd2'],
                's': measures['s'],
                'sd1/sd2': measures['sd1/sd2'],
                'breathingrate': measures['breathingrate'],
            }

            for feature_name, feature_value in hrv_features.items():
                data_current_patient.at[index, feature_name] = feature_value

        except Exception as e:
            print(f'Error processing PPG signal: {e}')
            continue

    df = df.append(data_current_patient)

df.to_csv('output_data/hp-processed_ppg_one_minute_8.csv', index=False)
