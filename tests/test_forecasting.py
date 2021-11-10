from taug.forecasters.deep import LSTMCNNForecaster
import numpy as np

X = np.random.normal(size=(100, 8, 10))
y = np.random.normal(size=(100, 4, 10))

model = LSTMCNNForecaster(8, 4, 10)
model.construct_forecaster()
model.fit(X, y)
print(model.predict(X).shape)