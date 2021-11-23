from augmentts.forecasters.deep import LSTMCNNForecaster
from augmentts.utils import prepare_ts
import numpy as np

ts = np.random.rand(100, 10)
X, y = prepare_ts(ts, 8, 4)

print(X.shape)
print(y.shape)

model = LSTMCNNForecaster(window_size=8, steps_ahead=4, n_series=10)
model.fit(X, y, epochs=10)
print(model.predict(X).shape)