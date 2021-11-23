import numpy as np

# preparing time series for rolling window regression
def prepare_ts(ts, n_steps_in, n_steps_out):
    """
    Prepare time series for rolling window regression.
    input parameters:
        ts: time series with shape (series_length, number_of_series)
        n_steps_in: number of steps in
        n_steps_out: number of steps out
    """
    X, y = list(), list()
    for i in range(len(ts)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(ts):
            break
        seq_x, seq_y = ts[i:end_ix, :], ts[end_ix:out_end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)