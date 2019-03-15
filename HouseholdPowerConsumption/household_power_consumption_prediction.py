#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 12:38:03 2019

@author: asal
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from keras import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def mean_absolute_percentage_error(y_true, y_pred):
	mapes = []
	horizons = y_true.shape[1]
	for h in range(horizons):
		mape = np.mean(np.abs((y_true[:,h] - y_pred[:,h]) / y_true[:,h])) * 100
		mapes.append(mape)
	total_mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
	return mapes, total_mape	   
	

def fill_missing_values(values):
	one_day_back_index = 24 * 60
	means = np.nanmean(values, axis=0)
	(rows, cols) = values.shape
	for row in range(rows):
		for col in range(cols):
			if np.isnan(values[row, col]):
				if (row - one_day_back_index) < 0:
					imputed_value = means[col]
				else:
					imputed_value = values[row - one_day_back_index, col] 
				values[row, col] = imputed_value
	return


def clean_transform_save_dataset():
		# Load the data
	dataset = pd.read_csv('household_power_consumption.txt', sep=';', 
					      header=0, low_memory=False, infer_datetime_format=True, 
						  parse_dates={'datetime':[0,1]}, index_col=['datetime'])

	# Replace '?' values with np.NaN
	dataset.replace('?', np.NaN, inplace=True)
	
	# Replace the dtype of the numerical variables of the dataset with float32
	dataset = dataset.astype('float32')
	
	# Fill missing values
	fill_missing_values(dataset.values)
	
	# Create new column, the remainder of the sub-metering
	values = dataset.values
	dataset['sub_metering_4'] = (values[:, 0] * 1000.0 / 60.0) - (values[:, 4] + values[:, 5] + values[:, 6])

	# Save new cleaned/tranformed dataset to .csv file
	dataset.to_csv('household_power_consumption.csv')
	
	return


def evaluate_multistep_forecasting(y, y_hat):
	rmses = []
	(num_samples, horizons) = y.shape
	for h in range(horizons):
		rmses.append(sqrt(mean_squared_error(y[:, h], y_hat[:, h])))
	total_rmse = 0.0
	for sample in range(num_samples):
		for h in range(horizons):
			total_rmse += ((y[sample, h] - y_hat[sample, h])**2)
	total_rmse = sqrt(total_rmse / float(num_samples * horizons))
	return rmses, total_rmse


def train_test_split(values, train_perc):
	(rows,) = values.shape
	train_index = int(train_perc * rows)
	train, test = values[0:train_index], values[train_index:rows]
	return train, test
	

def transform_to_ml_dataset_univariate(values, p, h):
	l = len(values)
	X = []
	Y = []
	for i in range(p, l - h + 1):
		# input vectors
		x = []
		for j in range(i - p, i):
			x.append(values[j])
		X.append(x)
		
		# output vectors
		y = []
		for j in range(i, i + h):
			y.append(values[j])
		Y.append(y)
	X = np.array(X)
	Y = np.array(Y)
	return X, Y

	
def build_vanilla_LSTM_model(p, h):
	model = Sequential()
	# default activation function for LSTM in Keras > 2.0 is linear
	model.add(LSTM(128, activation='tanh', input_shape=(1, p)))
	#model.add(Dense(4, activation='relu'))
	model.add(Dense(h))
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model	
	

def build_stacked_LSTM_model(p, h):
	model = Sequential()
	# default activation function for LSTM in Keras > 2.0 is linear
	model.add(LSTM(128, activation='tanh', return_sequences=True, input_shape=(1, p)))
	model.add(LSTM(64, activation='tanh'))
	#model.add(Dense(4, activation='relu'))
	model.add(Dense(h))
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model	


def build_baseline_model():
	model = LinearRegression()
	return model


def downsample(values, window_size):
	samples = np.split(values, values.shape[0]/window_size)
	values_downsampled = [np.mean(item) for item in samples]
	values_downsampled = np.array(values_downsampled)
	return values_downsampled


def extrapolate_values(values, n):
	l = len(values)
	values_extrapolated = []
	for i in range(l):
		temp = n * [values[i]]
		values_extrapolated.append(temp)
	values_extrapolated = np.array(values_extrapolated)
	values_extrapolated = values_extrapolated.flatten()
	return values_extrapolated


def main():
	# Reproducibility
	np.random.seed(42)
	
	# Whether to use baseline model or LSTM
	BASELINE = False
	
	# Load data
	dataset = pd.read_csv('household_power_consumption.csv')
	values = dataset.values
	del dataset
	
	# Keep a small portion of the data and 1 variable
	num_months = 3
	months_data_index = num_months * 30 * 24 * 60
	var_index = 1
	values = values[0:months_data_index, var_index]
	
	# Downsample data, i.e. aggregated minute values into 15-min intervals
	num_minutes_per_interval = 15
	values = downsample(values, num_minutes_per_interval)
	
	# Normalize data into [0,1] interval
	scaler = MinMaxScaler(feature_range=(0,1))
	values_normalized = scaler.fit_transform(values.reshape(-1,1))
	values_normalized = values_normalized.reshape(values.shape)
	
	# Split data into train and test
	train_perc = 0.8
	train, test = train_test_split(values_normalized, train_perc)
	
	# Transform data into an ML-compatible form
	num_previous_days = 1
	p = num_previous_days * int(24 * 60 / num_minutes_per_interval)
	h = int(24 * 60 / num_minutes_per_interval)
	X_train, Y_train = transform_to_ml_dataset_univariate(train, p, h)
	X_test, Y_test = transform_to_ml_dataset_univariate(test, p, h)
	
	#if BASELINE:
	model_LR = build_baseline_model()
	model_LR.fit(X_train, Y_train)
	train_predictions_LR = model_LR.predict(X_train)
	test_predictions_LR = model_LR.predict(X_test)
	#else:
		# Transform data into a form compatible with LSTM, i.e. [samples, timesteps, features]
	X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
	X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
	
	model_LSTM = build_vanilla_LSTM_model(p, h)
	model_LSTM.fit(X_train, Y_train, epochs=100, batch_size=1)
	train_predictions_LSTM = model_LSTM.predict(X_train)
	test_predictions_LSTM = model_LSTM.predict(X_test)
	
	# Make predictions on train and test data
	#train_predictions = model.predict(X_train)
	#test_predictions = model.predict(X_test)
	
	train_predictions = np.zeros(train_predictions_LR.shape)
	test_predictions = np.zeros(test_predictions_LR.shape)
	
	short_long_boundary = 4
	
	train_predictions[:, 0:short_long_boundary] = ((train_predictions_LR[:, 0:short_long_boundary] + train_predictions_LSTM[:, 0:short_long_boundary]) / 2.0)
	train_predictions[:, short_long_boundary:] = ((train_predictions_LR[:, short_long_boundary:] + train_predictions_LSTM[:, short_long_boundary:]) / 2.0)
	del train_predictions_LR
	del train_predictions_LSTM
	
	test_predictions[:, 0:short_long_boundary] = ((test_predictions_LR[:, 0:short_long_boundary] + test_predictions_LSTM[:, 0:short_long_boundary]) / 2.0)
	test_predictions[:, short_long_boundary:] = ((test_predictions_LR[:, short_long_boundary:] + test_predictions_LSTM[:, short_long_boundary:]) / 2.0)
	del test_predictions_LR
	del test_predictions_LSTM
	
	# Scale data back to origin range of values
	Y_train_scaled_back = scaler.inverse_transform(Y_train)
	train_predictions_scaled_back = scaler.inverse_transform(train_predictions)	
	Y_test_scaled_back = scaler.inverse_transform(Y_test)
	test_predictions_scaled_back = scaler.inverse_transform(test_predictions)
	
	# Compute and print error metrics on train and test data
	#train_rmse = sqrt(mean_squared_error(Y_train_scaled_back, train_predictions_scaled_back))
	#test_rmse = sqrt(mean_squared_error(Y_test_scaled_back, test_predictions_scaled_back))
	
	train_mapes, train_total_mape = mean_absolute_percentage_error(Y_train_scaled_back, train_predictions_scaled_back)
	test_mapes, test_total_mape = mean_absolute_percentage_error(Y_test_scaled_back, test_predictions_scaled_back)
	
#	print('Total Train RMSE (in kWh): ', round(train_rmse, 3))
#	print('Total Test RMSE (in kWh): ', round(test_rmse, 3))
#	print('\n')

	print('Total Train MAPE (%): ', round(train_total_mape, 3))
	print('Total Test MAPE (%): ', round(test_total_mape, 3))
	
	print('Train MAPE (%) for h = 1: ', round(train_mapes[0], 3))
	print('Test MAPE (%) for h = 1: ', round(test_mapes[0], 3))
	
	print('Train MAPE (%) for h = {}: '.format(h), round(train_mapes[h-1], 3))
	print('Test MAPE (%) for h = {}: '.format(h), round(test_mapes[h-1], 3))
	
	plt.figure(figsize=(10,5))
	plt.plot(train_mapes, label='train MAPEs')
	plt.plot(test_mapes, label='test MAPEs')
	plt.legend()
	plt.show()
	
	plt.figure(figsize=(10,5))
	plt.plot(Y_test_scaled_back[:,0], label='real')
	plt.plot(test_predictions_scaled_back[:,0], label='predictions')
	plt.title('h = 1')
	plt.legend()
	plt.show()
	
	plt.figure(figsize=(10,5))
	plt.plot(Y_test_scaled_back[:,h-1], label='real')
	plt.plot(test_predictions_scaled_back[:,h-1], label='predictions')
	plt.title('h = {}'.format(h))
	plt.legend()
	plt.show()
	
	
if __name__ == '__main__':
	main()