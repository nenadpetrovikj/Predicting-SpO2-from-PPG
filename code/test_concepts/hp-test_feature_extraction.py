## Imports

import pandas as pd
import numpy as np

## Load hp-data_real_ppg_representation.csv from output_data

data = pd.read_csv('output_data/output_data_large/hp-data_real_ppg_representation/hp-data_real_ppg_representation.csv')
print(data['PPG'])

## Convert PPG string lists into NumPy arrays

print(type(data.loc[1, 'PPG']))

data['PPG'] = data['PPG'].apply(lambda x: np.fromstring(x, sep=","))

print(type(data.loc[1, 'PPG']))

print(data['PPG'])

## Check for NaN values in the interpolated PPG signals

flag = False
for index, item in data.iterrows():
    ppg_signal = item['PPG']
    if np.isnan(ppg_signal).any():
        flag = True
        print(index, True)

if not flag:
    print(f"PPG signals don't contain NaN's")

## Conclusion - even though we have interpolated the PPG signals, some HeartPy extracted features have NaN, np.inf or -np.inf values

data.drop(columns=['ID', 'PPG', 'SpO2', 'interpolation_flag', 'SourceFile'], inplace=True)
inf_counts = (np.isinf(data)).sum()
missing_values = data.isnull().sum() + inf_counts
percentage = missing_values / len(data) * 100
missing_values_table = pd.DataFrame()
missing_values_table['Num. of np.inf or -np.inf values'] = inf_counts
missing_values_table['Num. of NaN values'] = missing_values - inf_counts
missing_values_table['Total num. of missing values'] = missing_values
missing_values_table['% of missing values'] = percentage
print(missing_values_table)