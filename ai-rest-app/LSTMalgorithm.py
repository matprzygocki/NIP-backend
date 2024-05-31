from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential


def create_model(test_x, test_y, train_x, train_y):
    model = Sequential()
    model.add(LSTM(10, input_shape=(1, 1)))
    model.add(Dense(1))
    model.add(Dropout(0.03))
    model.compile(loss='mean_squared_error', optimizer='adam')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    model.fit(train_x, train_y, epochs=100, batch_size=3, verbose=5, callbacks=[es], validation_data=(test_x, test_y))
    return model
