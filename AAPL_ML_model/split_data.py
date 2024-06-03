import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the data
data = pd.read_csv('AAPL.csv', index_col='timestamp', parse_dates=True)

# Select features and target
features = data[['open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions']]
target = data['close']

# Split the data into training and test sets (80% training, 20% test)
train_size = int(len(features) * 0.8)
train_features = features[:train_size]
test_features = features[train_size:]
train_target = target[:train_size]
test_target = target[train_size:]

# Scale the features
scaler = MinMaxScaler()
train_features_scaled = scaler.fit_transform(train_features)
test_features_scaled = scaler.transform(test_features)

# Prepare the data for PyTorch
window_size = 60

def create_sequences(features, target, window_size):
    X = []
    y = []
    for i in range(window_size, len(features)):
        X.append(features[i-window_size:i])
        y.append(target.iloc[i])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_features_scaled, train_target, window_size)
X_test, y_test = create_sequences(test_features_scaled, test_target, window_size)

# Save the training and test data
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)
