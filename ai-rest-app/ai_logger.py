import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def print_mean_squared_error(test_predict, test_y, train_predict, train_y):
    train_score = np.sqrt(mean_squared_error(train_y[0], train_predict[:, 0]))
    test_score = np.sqrt(mean_squared_error(test_y[0], test_predict[:, 0]))
    print('Train Score: %.2f RMSE' % train_score)
    print('Test Score: %.2f RMSE' % test_score)
    return train_score, test_score


def print_r2_score(test_predict, test_y, train_predict, train_y):
    train_r2 = r2_score(train_y[0], train_predict[:, 0])
    test_r2 = r2_score(test_y[0], test_predict[:, 0])
    print(f'Train R^2: {train_r2:.2f}')
    print(f'Test R^2: {test_r2:.2f}')
    return train_r2, test_r2
