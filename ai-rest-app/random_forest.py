from sklearn.ensemble import RandomForestRegressor


def create_model(test_x, test_y, train_x, train_y):
    model = RandomForestRegressor(n_estimators=100, random_state=10)
    model.fit(train_x, train_y)
    return model
