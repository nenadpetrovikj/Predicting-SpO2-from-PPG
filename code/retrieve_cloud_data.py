## Imports

import pandas as pd
import numpy as np
import owncloud
import pickle
import os
import heartpy as hp
import warnings
from pathlib import Path
from dotenv import load_dotenv
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")

## Load owncloud username and password from .env file

print("Working directory:", os.getcwd())

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)
username = os.getenv('OWNCLOUD_USERNAME')
password = os.getenv('OWNCLOUD_PASSWORD')

## File summary

def file_summary(data, file_path):
    signals_with_nan = 0
    signals_with_nan_5pc = 0
    spo2_below_95 = 0
    spo2_below_95_with_nan_5pc = 0  # number of entries where spo2 < 95 and nan values <= 5% of the PPG signal

    for index, item in data.iterrows():
        ppg_signal = item['PPG']
        spo2 = item['SpO2']
        if np.isnan(ppg_signal).any():
            signals_with_nan += 1
            if np.isnan(ppg_signal).mean() <= 0.05:
                signals_with_nan_5pc += 1
                if spo2 < 95:
                    spo2_below_95_with_nan_5pc += 1
        if spo2 < 95:
            spo2_below_95 += 1

    print(f'------- FILE SUMMARY: {file_path} --------')
    print(f'Total number of entries: {data.shape[0]}')
    print(f'# of unique patients inside one file: {data["ID"].nunique()}')
    print(f'# of entries where SpO2 < 95: {spo2_below_95}')
    print(f'# of entries where SpO2 >= 95: {data.shape[0] - spo2_below_95}')
    print(f'# of entries where the PPG signal has at least one Nan: {signals_with_nan}')
    print(f'# of entries where the PPG signal has a maximum of 5% Nan: {signals_with_nan_5pc}')
    print(f'# of entries where the PPG signal has more than 5% Nan: {signals_with_nan - signals_with_nan_5pc}')

    if signals_with_nan != 0:
        print(f'% of entries where the PPG signal has at least one Nan [PPG-Including-Nan]: {(signals_with_nan/data.shape[0])*100:.3f}%')
        print(f'% of entries where the PPG signal has a maximum of 5% Nan, from PPG-Including-Nan: {(signals_with_nan_5pc/signals_with_nan)*100:.3f}%')
        print(f'% of entries where the PPG signal has more than 5% Nan, from PPG-Including-Nan: {(1-signals_with_nan_5pc/signals_with_nan)*100:.3f}%')
        print(f'% of entries where SpO2 < 95, from PPG-Including-Nan (max 5%): {(spo2_below_95_with_nan_5pc/signals_with_nan_5pc)*100:.3f}%')

## Interpolation and deletion of entries in a file

def file_processing(data):
    data['interpolation_flag'] = 0
    for index, item in data.iterrows():
        ppg_signal = item['PPG']
        if np.isnan(ppg_signal).any():
            if np.isnan(ppg_signal).mean() <= 0.05:
                valid_indices = np.arange(len(ppg_signal))[~np.isnan(ppg_signal)]
                interpolated_values = np.interp(np.arange(len(ppg_signal)), valid_indices, ppg_signal[valid_indices])
                data.at[index, 'PPG'] = interpolated_values
                data.at[index, 'interpolation_flag'] = 1
            else:
                data = data.drop(index)

    print(f'------- FILE PROCESSING DONE --------')
    processed_data = data
    return processed_data

## Return selected rows for patient

def get_measurements(group):
    data = pd.DataFrame(columns=group.columns)
    for index, item in group.iterrows():
        # item['PPG'] = hp.filter_signal(item['PPG'], cutoff=[0.5, 3.75], sample_rate=125, order=4, filtertype='bandpass')
        spo2 = item['SpO2']
        if spo2 < 95:
            data = data.append(item, ignore_index=True)
            group.drop(index, inplace=True)

    group_size = group.shape[0]
    data_size = data.shape[0]

    if data_size < group_size:  # more data with SpO2 >= 95
        rows_to_be_selected = data_size
    else:                       # more data with SpO2 < 95
        rows_to_be_selected = group_size

    random_rows = group.sample(n=rows_to_be_selected, replace=False)
    data = data.append(random_rows, ignore_index=True)

    return data

## Extracting HeartPy features

def feature_extraction_hp(data, file_path):
    extracted_data = pd.DataFrame()

    data['SourceFile'] = file_path

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

        extracted_data = extracted_data.append(data_current_patient, ignore_index=True)

    print(f'------- FEATURE EXTRACTION DONE --------')
    return extracted_data

## Extracting Respiratory features

def feature_extraction_respiratory(data, file_path):
    extracted_data = pd.DataFrame()

    data['SourceFile'] = file_path

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

        extracted_data = extracted_data.append(data_current_patient, ignore_index=True)

    print(f'------- FEATURE EXTRACTION DONE --------')
    return extracted_data

## Final dataset summary
# Conclusion - The distribution might be skewed, since there are more entries where SpO2 < 95

def dataset_summary(data):
    print(f'------- DATASET SUMMARY --------')
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    nan_mask = data[data.isna().any(axis=1)]
    print(f'(Rows, columns) = {data.shape}')
    print(f"# of PPG signals with Sp02 < 95: {(data['SpO2'] < 95).sum()}")
    print(f"# of PPG signals with Sp02 >= 95: {(data['SpO2'] >= 95).sum()}")
    print(f"Interpolation applied in {data['interpolation_flag'].sum()} PPG signals")
    print(f"Feature extraction failed in {data.isna().any(axis=1).sum()} entries")
    print(f"Out of entries with failed feature extraction, interpolated are: {len(nan_mask[nan_mask['interpolation_flag'] == 1])} entries")
    print(f"# of patients - taking into account overlap in files [REAL]: {data['ID'].nunique()}")

## Make a request, retrieve the data, clean it, select specific entries for patients, perform feature extraction and save the new database
# Time taken to process files in Google Collab:
# 1. Files <= 250MB (respiratory): 20m
# 2. Files <= 350MB (respiratory): 40m
# 3. Files <= 400MB (respiratory): CRASH
# 4. Files <= 250MB (hp): 45m
# 5. Files <= 350MB (hp): 1h 52m

client = owncloud.Client('https://sp4life.finki.ukim.mk')
client.login(username, password)

folder_path = 'Timski_Proekt_prof.Koteska/PPG_One_Minute'

file_list = [file_ for file_ in client.list(folder_path) if file_.get_size() <= 262144000]
print(f'Number of selected files: {len(file_list)}')

final_extracted_data = pd.DataFrame()

for file_ in file_list:
    file_path = file_.path
    file_contents = client.get_file_contents(file_path)
    unpickled_data = pickle.loads(file_contents)

    file_summary(unpickled_data, file_path)
    processed_data = file_processing(unpickled_data)

    extracted_data = feature_extraction_hp(processed_data, file_path)
    final_extracted_data = final_extracted_data.append(extracted_data, ignore_index=True)

# Save .csv where PPG signal is a real representation of measurements in a list, instead of a flattened string
# print(final_extracted_data.loc[1, 'PPG'])
# final_extracted_data['PPG'] = final_extracted_data['PPG'].apply(lambda x: ','.join(map(str, x)))
# print(final_extracted_data.loc[1, 'PPG'])

final_extracted_data.to_csv('output_data/data_extraction_test.csv', index=False)

data = pd.read_csv('output_data/output_data_large/respiratory-data_extracted_filter_600MB.csv')
# Run the following line to remove outliers
# data.drop(index=data[data['SpO2'] < 80].index, inplace=True)
dataset_summary(data)
