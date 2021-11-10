import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
import tensorflow_addons as tfa


class DeepForecaster():
    """
    Base class for deep forecasters
    """
    def __init__(self, *args, **kwargs):
        self.forecaster = None
        
    
    def construct_forecaster(self, *args, **kwargs):
        raise NotImplementedError("The forecaster is not implemented. You must provide an implementation for construct_forecaster method")
        

class LSTMCNNForecaster(DeepForecaster):
    """
    LSTM-CNN Forecaster Class
    """
    def __init__(self, window_size, steps_ahead, n_series):
        super().__init__()
        self.steps_ahead = steps_ahead
        self.n_series = n_series
        self.window_size = window_size
        self.construct_forecaster()
        self.fit = self.forecaster.fit
        self.predict = self.forecaster.predict

    def construct_forecaster(self):
        input_model = tf.keras.Input(shape=(self.window_size, self.n_series))
        x = tf.keras.layers.GaussianNoise(0.01)(input_model)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, activation='tanh', use_bias=True))(x)
        x = tf.keras.layers.RepeatVector(14)(x)
        x = tf.keras.layers.Dropout(0.9)(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, activation='tanh', return_sequences=False, use_bias=True))(x)
        x = tf.keras.layers.RepeatVector(self.window_size)(x)
        x = tf.keras.layers.Dropout(0.9)(x)
        x = tf.keras.layers.Concatenate(axis=2)([input_model, x])
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, activation='tanh', return_sequences=False, use_bias=True))(x)
        x = tf.keras.layers.RepeatVector(self.steps_ahead)(x)
        x = tf.keras.layers.Dropout(0.9)(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, activation='tanh', return_sequences=True, use_bias=True))(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Conv1D(64, 1, 1, activation='relu', use_bias=True)(x)

        output_model = tf.keras.layers.Conv1D(self.n_series, 1, 1, activation='relu', use_bias=True)(x)

        final_model = tf.keras.Model(input_model, output_model)
        final_model.compile(optimizer='adam', loss='mae', metrics=['mse',tf.keras.metrics.RootMeanSquaredError(),
                                                    'mae', 'msle', tf.keras.losses.mean_absolute_percentage_error,
                                                    tf.keras.losses.huber, tf.keras.losses.log_cosh])
        self.forecaster = final_model

