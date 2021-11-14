from taug.forecasters.deep import LSTMCNNForecaster
import numpy as np

X = np.random.normal(size=(100, 8, 10))
y = np.random.normal(size=(100, 4, 10))

model = LSTMCNNForecaster(window_size=8, steps_ahead=4, n_series=10)
model.fit(X, y, epochs=10)
print(model.predict(X).shape)