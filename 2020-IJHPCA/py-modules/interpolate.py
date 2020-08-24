import numpy as np

import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def interpolate_real_data(X, Y, degree=2):
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    regression = LinearRegression().fit(X_poly, Y)
    X_range = np.arange(X.min(), X.max() + 1)
    predicted_ys = regression.predict(poly.fit_transform(X_range.reshape(-1, 1)))
    if predicted_ys.ndim == 2:
        predicted_ys = predicted_ys.reshape(predicted_ys.shape[0])
    return X_range, predicted_ys
