import random
from typing import BinaryIO
import random_forest as rforest
import numpy as np
import tensorflow as tf
import LSTMalgorithm as lstm
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler


import ai_logger as log

SEED = 7


def predict(path_or_content: str | BinaryIO, split_percentage, algorithm):
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = read_dataset(path_or_content, scaler)
    return predict_from_dataset(dataset, scaler, split_percentage, algorithm)



def predict_from_dataset(dataset, scaler, split_percentage, algorithm):
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    test, train = divide_dataset_into_training_and_test_data(dataset, split_percentage, algorithm)

    # reshape into X=t and Y=t+1
    look_back = 1
    train_x, train_y = offset_datasets_by_one(train)
    test_x, test_y = offset_datasets_by_one(test)

    # reshape input to be [samples, time steps, features]
    train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
    test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))

    train_x_rf = train_x.reshape(-1, 1)
    test_x_rf = test_x.reshape(-1, 1)

    match(algorithm):
        case 1:
            model = lstm.create_model(test_x, test_y, train_x, train_y)
            # make predictions
            train_predict = model.predict(train_x)
            test_predict = model.predict(test_x)
        case 2:
            # Użyj spłaszczonych danych dla algorytmu Random Forest

            model = rforest.create_model(test_x_rf, test_y, train_x_rf, train_y)
            # make predictions
            train_predict = model.predict(train_x_rf)
            test_predict = model.predict(test_x_rf)


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


def divide_dataset_into_training_and_test_data(dataset, split_percentage, alg):
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
