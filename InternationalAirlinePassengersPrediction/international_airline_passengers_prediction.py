import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import math


Train_Perc = 0.67
P = 3
Epochs = 20
Batch_Size = 1


def first_differences(ts):
	l = len(ts)
	tf_first_diff = ts[1:] - ts[0:(l - 1)]
	return tf_first_diff


def train_test_split_ts(ts, train_perc):
	l = len(ts)
	train_index = int(train_perc * l)
	train_ts, test_ts = ts[0:train_index, :], ts[train_index:l, :]
	return train_ts, test_ts


def transform_ts_to_ml_dataset(ts, p):
	l = len(ts)
	X = []
	y = []
	for i in range(p, l):
		x = []
		for j in range(i - p, i):
			x.append(ts[j])
		X.append(x)
		y.append(ts[i])
	X = np.array(X)
	X = X.reshape((X.shape[0], X.shape[1]))
	y = np.array(y)
	return X, y


def build_LSTM_model(p):
	model = Sequential()
	model.add(LSTM(4, input_shape=(1, p)))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model


def main():
	dataset = pd.read_csv('international-airline-passengers.csv', usecols=[1], skipfooter=3)
	dataset.columns = ['passengers']
	dataset = dataset.values # data in an ndarray
	dataset = dataset.astype('float32') # change the type of values from int64 to float32

	# Normalize data into the range of values [0, 1]
	scaler = MinMaxScaler(feature_range=(0, 1))
	dataset_normalized = scaler.fit_transform(dataset)

	# Split timeseries into train and test
	train_ts, test_ts = train_test_split_ts(dataset_normalized, Train_Perc)

	# Transform timeseries data into an ML-compatible form
	X_train, y_train = transform_ts_to_ml_dataset(train_ts, P)
	X_test, y_test = transform_ts_to_ml_dataset(test_ts, P)

	# Reshape data into a form compatible with the LSTM units, i.e. [samples, time steps, features]
	X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
	X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

	# Build LSTM model
	model = build_LSTM_model(P)
	model.fit(X_train, y_train, epochs=Epochs, batch_size=Batch_Size)

	# Make predictions on train and test data
	train_predictions = model.predict(X_train)
	test_predictions = model.predict(X_test)

	# Scale data back to origin range of values
	y_train_scaled_back = scaler.inverse_transform(y_train)
	train_predictions_scaled_back = scaler.inverse_transform(train_predictions)
	y_test_scaled_back = scaler.inverse_transform(y_test)
	test_predictions_scaled_back = scaler.inverse_transform(test_predictions)

	# Compute and print error metrics on train and test data
	train_rmse = math.sqrt(mean_squared_error(y_train_scaled_back, train_predictions_scaled_back))
	test_rmse = math.sqrt(mean_squared_error(y_test_scaled_back, test_predictions_scaled_back))

	print('Train RMSE (x1000 passengers): ', round(train_rmse, 3))
	print('Test RMSE (x1000 passengers): ', round(test_rmse, 3))


if __name__ == '__main__':
	main()