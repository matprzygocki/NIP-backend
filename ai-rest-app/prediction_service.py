import random
from typing import BinaryIO

import numpy as np
import tensorflow as tf
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential

import ai_logger as log

SEED = 7


def predict(path_or_content: str | BinaryIO, split_percentage):
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = read_dataset(path_or_content, scaler)
    return predict_from_dataset(dataset, scaler, split_percentage)


def predict_from_dataset(dataset, scaler, split_percentage):
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    test, train = divide_dataset_into_training_and_test_data(dataset, split_percentage)

    # reshape into X=t and Y=t+1
    look_back = 1
    train_x, train_y = offset_datasets_by_one(train)
    test_x, test_y = offset_datasets_by_one(test)

    # reshape input to be [samples, time steps, features]
    train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
    test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))

    model = create_model(test_x, test_y, train_x, train_y)

    # make predictions
    train_predict = model.predict(train_x)
    test_predict = model.predict(test_x)

    # invert predictions
    train_predict = scaler.inverse_transform(train_predict)
    train_y = scaler.inverse_transform([train_y])
    test_predict = scaler.inverse_transform(test_predict)
    test_y = scaler.inverse_transform([test_y])

    train_predict_plot = shift_results(dataset, look_back, len(train_predict) + look_back, train_predict)
    test_predict_plot = shift_results(dataset, len(train_predict) + (look_back * 2) + 1, len(dataset) - 1, test_predict)

    train_score, test_score = log.print_mean_squared_error(test_predict, test_y, train_predict, train_y)
    train_r2, test_r2 = log.print_r2_score(test_predict, test_y, train_predict, train_y)

    return {
        "train": {
            "x": list(map(lambda arr: arr[0][0], train_x.tolist())),
            "y": train_predict_plot[~np.isnan(train_predict_plot)].tolist()
        },
        "test": {
            "x": list(map(lambda arr: arr[0][0], test_x.tolist())),
            "y": test_predict_plot[~np.isnan(test_predict_plot)].tolist()
        },
        "meanSquaredError": {
            "train": train_score,
            "test": test_score
        },
        "r2": {
            "train": train_r2,
            "test": test_r2
        }
    }


def shift_results(dataset, begin, end, prediction):
    train_predict_plot = np.empty_like(dataset)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[begin:end, :] = prediction
    return train_predict_plot


def create_model(test_x, test_y, train_x, train_y):
    model = Sequential()
    model.add(LSTM(10, input_shape=(1, 1)))
    model.add(Dense(1))
    model.add(Dropout(0.03))
    model.compile(loss='mean_squared_error', optimizer='adam')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    model.fit(train_x, train_y, epochs=100, batch_size=3, verbose=5, callbacks=[es], validation_data=(test_x, test_y))
    return model


def divide_dataset_into_training_and_test_data(dataset, split_percentage):
    random.shuffle(dataset)
    train_size = int(len(dataset) * split_percentage)
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    return test, train


def read_dataset(path_or_file: BinaryIO | str, scaler):
    dataframe = read_csv(path_or_file, sep=';')
    bitcoin_prices = dataframe['Open']
    dataset = bitcoin_prices.values
    dataset = dataset.astype('float32')
    dataset = scaler.fit_transform(dataset.reshape(-1, 1))
    return dataset


def offset_datasets_by_one(dataset):
    data_x, data_y = [], []
    for i in range(len(dataset) - 1 - 1):
        a = dataset[i:(i + 1), 0]
        data_x.append(a)
        data_y.append(dataset[i + 1, 0])
    return np.array(data_x), np.array(data_y)
