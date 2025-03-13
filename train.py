import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, Dense

# ========================== #
#  Step 1: Load & Preprocess Data
# ========================== #

SEQ_LEN = 1 

def read_csv(file_path):
    """
    Reads a CSV file and returns a dictionary mapping (measure, tool) to a list of numeric values.
    Expected header: ProgName, Cycle, Tool, Block, MeasureType, Timestamp, Value
    """
    valid_measures = {"POWER", "VIB_VELO_X", "VIB_VELO_Y", "VIB_VELO_Z",
                      "VIB_ACCEL_X", "VIB_ACCEL_Y", "VIB_ACCEL_Z"}
    
    data_dict = {}
    
    with open(file_path, 'r') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(',')
            if len(parts) in [7, 8]:  
                tool = parts[2].strip()
                measure_candidate = parts[4].strip() if parts[4].strip() in valid_measures else parts[5].strip()
                if measure_candidate not in valid_measures:
                    continue
                try:
                    value = float(parts[-1].strip())
                except ValueError:
                    continue
                key = (measure_candidate, tool)
                data_dict.setdefault(key, []).append(value)
    return data_dict


def process_directory(directory):
    """ Processes all CSV files in a directory (each file is one cycle) and returns structured data. """
    cycle_data = {}
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            cycle = filename.split('.')[0].strip()  # e.g., "438"
            file_path = os.path.join(directory, filename)
            cycle_data[cycle] = read_csv(file_path)
    return cycle_data


def load_data(directory):
    """ Loads all cycles and structures data as (measure, tool) -> time series per cycle """
    cycle_data = process_directory(directory)
    time_series_dict = defaultdict(list)
    
    for cycle in sorted(cycle_data.keys(), key=int):
        for (measure, tool), values in cycle_data[cycle].items():
            if values:  # Ensure there is data to aggregate
                time_series_dict[(measure, tool)].append(np.mean(values))  # Aggregate per cycle
    
    return {key: np.array(val) for key, val in time_series_dict.items() if len(val) > SEQ_LEN}


# Load normal and abnormal datasets
normal_data = load_data("normal_csv")
abnormal_data = load_data("abnormal_csv")

# ========================== #
#  Step 2: Prepare Data for LSTM
# ========================== #

def create_sequences(data_dict, seq_length):
    """ Convert dictionary of time-series into LSTM-compatible sequences """
    sequences = []
    labels = []
    
    for key, values in data_dict.items():
        values = np.array(values)
        if len(values) > seq_length:
            for i in range(len(values) - seq_length):
                sequences.append(values[i:i+seq_length].reshape(seq_length, 1))  # Ensure correct shape
                labels.append(values[i+seq_length])
    
    return np.array(sequences), np.array(labels)


# Normalize data
scaler = MinMaxScaler()
for key in normal_data:
    normal_data[key] = scaler.fit_transform(normal_data[key].reshape(-1, 1)).flatten()

# Create training sequences
X_train, y_train = create_sequences(normal_data, SEQ_LEN)

# Debugging: Check shape before reshaping
print("X_train shape before reshaping:", X_train.shape)

if X_train.shape[0] == 0:
    print("Error: Not enough sequential data. Try reducing SEQ_LEN or check normal_csv files.")
    exit()

# Reshape correctly for LSTM input
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))  # Ensure 3D shape

# ========================== #
#  Step 3: Train LSTM-VAE Model
# ========================== #

timesteps, features = X_train.shape[1], X_train.shape[2]

inputs = Input(shape=(timesteps, features))
encoded = LSTM(64, activation='relu', return_sequences=True)(inputs)
encoded = LSTM(32, activation='relu', return_sequences=False)(encoded)

decoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(32, activation='relu', return_sequences=True)(decoded)
decoded = LSTM(64, activation='relu', return_sequences=True)(decoded)
decoded = Dense(features, activation='linear')(decoded)

vae = Model(inputs, decoded)
vae.compile(optimizer='adam', loss='mse')

# Train the model
vae.fit(X_train, X_train, epochs=50, batch_size=32, validation_split=0.1, shuffle=True)

# ========================== #
#  Step 4: Detect Anomalies in abnormal_csv
# ========================== #

# Scale abnormal data
for key in abnormal_data:
    abnormal_data[key] = scaler.transform(abnormal_data[key].reshape(-1, 1)).flatten()

X_test, _ = create_sequences(abnormal_data, SEQ_LEN)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Get predictions
y_pred = vae.predict(X_test)

# Compute reconstruction error
mse = np.mean(np.abs(y_pred - X_test), axis=(1,2))

# Identify threshold (99th percentile of training errors) to reduce false positives
threshold = np.percentile(mse, 99)

# Find the tool with the **highest** anomaly score
tool_errors = {key: mse[i] for i, key in enumerate(abnormal_data.keys())}
broken_tool = max(tool_errors, key=tool_errors.get)

print("Detected Broken Tool:", broken_tool)

# ========================== #
#  Step 5: Prove Tool Wear (Trend Analysis)
# ========================== #

if broken_tool in abnormal_data and len(abnormal_data[broken_tool]) > 1:
    cycles = list(range(len(abnormal_data[broken_tool])))
    plt.plot(cycles, abnormal_data[broken_tool], label=f"{broken_tool} (Abnormal)", color='red')

if broken_tool in normal_data and len(normal_data[broken_tool]) > 1:
    cycles = list(range(len(normal_data[broken_tool])))
    plt.plot(cycles, normal_data[broken_tool], linestyle='dashed', label=f"{broken_tool} (Normal)", color='blue')

plt.xlabel("Cycle")
plt.ylabel("Normalized Sensor Value")
plt.legend()
plt.title("Tool Degradation Over Cycles")
plt.show()
