import csv
import io
from typing import BinaryIO, Annotated

import numpy as np
import pandas as pd
import pandas.io.formats.csvs
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pandas import read_csv
from pandas._typing import ReadCsvBuffer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from starlette.middleware.cors import CORSMiddleware
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential


SEED = 7

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return "AI FastAPI"


@app.get("/health")
def health():
    return {"status": "UP"}


@app.post("/predict/{csv_file}", response_class=JSONResponse)
def post_predict_existing_file(csv_file, split_percentage: float = 0.67):
    return predict('./datasets/' + csv_file,  split_percentage)

@app.get("/predict/{csv_file}", response_class=JSONResponse)
def get_predict_existing_file(csv_file, split_percentage: float = 0.67):
    return predict('./datasets/' + csv_file,  split_percentage)

@app.post("/predict", response_class=JSONResponse)
async def post_predict(file: UploadFile = File(...), split_percentage: float = 0.67):
    path = f"./datasets/{file.filename}"
    return predict(path,  split_percentage)


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
    trainX, trainY = offset_datasets_by_one(train)
    testX, testY = offset_datasets_by_one(test)

    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    model = create_model(testX, testY, trainX, trainY)

    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    trainPredictPlot = shift_results(dataset, look_back, len(trainPredict) + look_back, trainPredict)
    testPredictPlot = shift_results(dataset, len(trainPredict) + (look_back * 2) + 1, len(dataset) - 1, testPredict)

    print_mean_squared_error(testPredict, testY, trainPredict, trainY)
    print_r2_score(testPredict, testY, trainPredict, trainY)

    result_map = {
        "train": {
            "x": list(map(lambda arr: arr[0][0], trainX.tolist())),
            "y": trainPredictPlot[~np.isnan(trainPredictPlot)].tolist()
        },
        "test": {
            "x": list(map(lambda arr: arr[0][0], testX.tolist())),
            "y": testPredictPlot[~np.isnan(testPredictPlot)].tolist()
        }
    }

    result_map_encoded = jsonable_encoder(result_map)
    return JSONResponse(content=result_map_encoded)


def shift_results(dataset, begin, end, prediction):
    train_predict_plot = np.empty_like(dataset)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[begin:end, :] = prediction
    return train_predict_plot


def print_mean_squared_error(testPredict, testY, trainPredict, trainY):
    # calculate root mean squared error
    trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))


def print_r2_score(testPredict, testY, trainPredict, trainY):
    # Calculate R^2 score for train and test
    train_r2 = r2_score(trainY[0], trainPredict[:, 0])
    test_r2 = r2_score(testY[0], testPredict[:, 0])
    print(f'Train R^2: {train_r2:.2f}')
    print(f'Test R^2: {test_r2:.2f}')


def create_model(testX, testY, trainX, trainY):
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(10, input_shape=(1, 1)))
    model.add(Dense(1))
    model.add(Dropout(0.03))
    model.compile(loss='mean_squared_error', optimizer='adam')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    model.fit(trainX, trainY, epochs=100, batch_size=3, verbose=5, callbacks=[es], validation_data=(testX, testY))
    return model


def divide_dataset_into_training_and_test_data(dataset, split_percentage):
    train_size = int(len(dataset) * split_percentage)
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    return test, train


def read_dataset(path: UploadFile | str, scaler):
    # load the dataset
    if type(path) is str:
        dataframe = read_csv(path, usecols=[1], engine='python')
    else:
        with open(path) as readfile:
            dataframe=pd.read_csv(readfile.read())
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    # normalize the dataset
    dataset = scaler.fit_transform(dataset)
    return dataset

# convert an array of values into a dataset matrix
def offset_datasets_by_one(dataset):
    dataX, dataY = [], []
    for i in range(len(dataset) - 1 - 1):
        a = dataset[i:(i + 1), 0]
        dataX.append(a)
        dataY.append(dataset[i + 1, 0])
    return np.array(dataX), np.array(dataY)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
