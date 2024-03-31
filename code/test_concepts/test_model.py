## Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout, Conv1D, MaxPooling1D, Flatten

## Load .csv file to use as a database in model creation

data = pd.read_csv('output_data/respiratory-data_extracted_filter_350MB.csv', on_bad_lines='skip')

## Prepare dataset for training - clean inf and Nan & drop unnecessary columns

data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)
data.drop(columns=['PPG', 'interpolation_flag', 'SourceFile'], inplace=True)
data.info()

## The dataset is split by patients in a train & test ratio ~80/20%. Each subset is then mixed up.

def train_test_split(data):

    data_groupedBy_patientID = data.groupby('ID')
    num_groups = len(data_groupedBy_patientID.groups)
    print(f'Number of patients in dataset: {num_groups}')

    group_indices = list(data_groupedBy_patientID.groups.keys())
    np.random.shuffle(group_indices)

    data_train = pd.DataFrame()
    data_test = pd.DataFrame()

    num_groups_processed = 0
    num_max_groups = int(num_groups * 0.80)

    for group_key in group_indices:
        group = data_groupedBy_patientID.get_group(group_key)

        if num_groups_processed >= num_max_groups:
            data_test = data_test.append(group, ignore_index=True)
        else:
            data_train = data_train.append(group, ignore_index=True)
        num_groups_processed += 1

    train_groups = data_train.groupby('ID')
    print(f'Number of patients in train dataset: {len(train_groups.groups)}')
    test_groups = data_test.groupby('ID')
    print(f'Number of patients in test dataset: {len(test_groups.groups)}')

    # Shuffle the rows of the dataset; frac=1 shuffles all rows, random_state for reproducibility
    data_train.drop(columns=['ID'], inplace=True)
    data_train = data_train.sample(frac=1, random_state=42)
    X_train = data_train.drop(columns=['SpO2'])
    y_train = data_train['SpO2']

    data_test.drop(columns=['ID'], inplace=True)
    data_test = data_test.sample(frac=1, random_state=42)
    X_test = data_test.drop(columns=['SpO2'])
    y_test = data_test['SpO2']

    return X_train, X_test, y_train, y_test

## Call train_test_split function

X_train, X_test, y_train, y_test = train_test_split(data)

## Build model

model = RandomForestRegressor(max_depth=50, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

## Accuracies

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R2: {r2}')

## Adjust test and pred sets for plotting

y_pred_flattened = y_pred.flatten()
y_pred_series = pd.Series(y_pred_flattened)

y_pred_series = y_pred_series.reindex(y_test.index)

y_test_sorted = y_test.sort_index()
y_pred_sorted = y_pred_series.sort_index()

## Plot results

plt.plot(y_test_sorted, label='Actual Data', color='blue')

# Plot predictions
plt.plot(y_pred_sorted, label='Predictions', color='red')

# Add legend
plt.legend()

# Add labels and title
plt.xlabel('Index')
plt.ylabel('SpO2 Level')
plt.title('Actual vs. Predicted SpO2 Levels')

# Show plot
plt.show()
