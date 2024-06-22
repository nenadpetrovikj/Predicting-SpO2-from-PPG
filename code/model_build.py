## Imports

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout, Conv1D, MaxPooling1D, Flatten

warnings.filterwarnings("ignore")

## Load .csv file to use as a database in model creation

data = pd.read_csv('output_data/output_data_large/respiratory-data_extracted_filter_600MB.csv', on_bad_lines='skip')
data.drop(index=data[data['SpO2'] < 80].index, inplace=True)  # drop all rows where the value for SpO2 < 80

data.info()

## Prepare dataset for training - clean inf and Nan & drop unnecessary columns

data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)
data.drop(columns=['PPG', 'interpolation_flag', 'SourceFile'], inplace=True)

data.info()

## View numer of patients and division of entries by an SpO2 level of 95

print(f"# of PPG signals with Sp02 < 95: {(data['SpO2'] < 95).sum()}")
print(f"# of PPG signals with Sp02 >= 95: {(data['SpO2'] >= 95).sum()}")
print(f"# of patients - taking into account overlap in files [REAL]: {data['ID'].nunique()}")

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

    print("Training set is " + str(int((len(X_test)/len(data)) * 100)) + "% of the whole dataset")

    return X_train, X_test, y_train, y_test

## Call train_test_split function

X_train, X_test, y_train, y_test = train_test_split(data)

## Define multiple ML models

# input_shape = (13, 1)  # hp features
input_shape = (10, 1)  # respiratory features

def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))  # Output layer for SpO2 prediction
    return model

def create_bilstm_model(input_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    return model

def create_rnn_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dense(1))
    return model

def create_rfr_model():
    model = RandomForestRegressor(max_depth=50, n_estimators=100, random_state=42, n_jobs=-1)
    return model

## Plotting function

def create_plot(model_name, y_pred, y_test):

    y_pred_flattened = y_pred.flatten()
    y_pred_series = pd.Series(y_pred_flattened)

    y_pred_series = y_pred_series.reindex(y_test.index)

    y_test_sorted = y_test.sort_index()
    y_pred_sorted = y_pred_series.sort_index()

    plt.figure()
    plt.plot(y_test_sorted, label='Actual Data', color='blue')

    # Plot predictions
    plt.plot(y_pred_sorted, label='Predictions', color='red')

    # Add legend
    plt.legend()

    # Add labels and title
    plt.xlabel('Index')
    plt.ylabel('SpO2 Level')
    plt.title(f'{model_name} - Actual vs. Predicted SpO2 Levels')

    plt.show()

## Build each model and save the results

models = {
    'LSTM': create_rnn_model(input_shape),
    'BiLSTM': create_bilstm_model(input_shape),
    'CNN': create_cnn_model(input_shape),
    'RFR': create_rfr_model()
}

results = pd.DataFrame(columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2'])

for model_name, model in models.items():
    if model_name == 'RFR':
        model.fit(X_train, y_train)
    elif model_name == 'BiLSTM':
        model.compile(optimizer='adam', loss='mean_absolute_error')
        model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=1)
    else:
        model.compile(optimizer='adam', loss='mean_absolute_error')
        model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    create_plot(model_name, y_pred, y_test)

    results = results.append({'Model': model_name, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}, ignore_index=True)

print(results)

results.to_csv('results/models_results.csv', index=False)

