import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from starlette.middleware.cors import CORSMiddleware
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential

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

@app.get("/predict", response_class=JSONResponse)
def predict():
    np.random.seed(7)

    # convert an array of values into a dataset matrix
    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    # fix random seed for reproducibility
    tf.random.set_seed(7)
    # load the dataset
    dataframe = read_csv('./datasets/airline-passengers.csv', usecols=[1], engine='python')
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # split into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    # reshape into X=t and Y=t+1
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(10, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.add(Dropout(0.03))
    model.compile(loss='mean_squared_error', optimizer='adam')

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    model.fit(trainX, trainY, epochs=100, batch_size=3, verbose=5, callbacks=[es], validation_data=(testX, testY))
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict

    # calculate root mean squared error
    trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))

    # Calculate R^2 score for train and test
    train_r2 = r2_score(trainY[0], trainPredict[:, 0])
    test_r2 = r2_score(testY[0], testPredict[:, 0])
    print(f'Train R^2: {train_r2:.2f}')
    print(f'Test R^2: {test_r2:.2f}')

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


@app.get("/predict2",response_class=JSONResponse)
def predict2():
    np.random.seed(7)
    def create_dataset2(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    # fix random seed for reproducibility
    tf.random.set_seed(7)
    # load the dataset
    dataframe = read_csv('./datasets/airline-passengers.csv', usecols=[1], engine='python')
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # split into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    # reshape into X=t and Y=t+1
    look_back = 1
    trainX, trainY = create_dataset2(train, look_back)
    testX, testY = create_dataset2(test, look_back)
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(10, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.add(Dropout(0.03))
    model.compile(loss='mean_squared_error', optimizer='adam')

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    model.fit(trainX, trainY, epochs=100, batch_size=3, verbose=5, callbacks=[es], validation_data=(testX, testY))
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict

    # calculate root mean squared error
    trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))

    # Calculate R^2 score for train and test
    train_r2 = r2_score(trainY[0], trainPredict[:, 0])
    test_r2 = r2_score(testY[0], testPredict[:, 0])
    print(f'Train R^2: {train_r2:.2f}')
    print(f'Test R^2: {test_r2:.2f}')

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

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
