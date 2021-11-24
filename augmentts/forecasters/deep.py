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
    Base class for deep forecasting models
    """
    def __init__(self, window_size, steps_ahead, loss='mae', optimizer='adam', metrics=None):
        
        self.steps_ahead = steps_ahead
        self.window_size = window_size
        self.loss = loss
        self.optimizer = optimizer
        
        if metrics is None:
            self.metrics = ['mae', 'mse']
        
        self.construct_forecaster()
        
        self.fit = self.forecaster.fit
        self.predict = self.forecaster.predict
        
    
    def construct_forecaster(self):
        final_model = self.forecaster_architecture()
        final_model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        self.forecaster = final_model


    def forecaster_architecture(self, X):
        raise NotImplementedError("The forecaster is not implemented. You must provide an implementation for forecaster_architecture method")

class LSTMCNNForecaster(DeepForecaster):
    """
    LSTM-CNN Forecaster Class
    """
    def __init__(self, n_series, lstm_hiddens=256, dropout_rate=0.9, lstm_activation='tanh', conv_activation='relu', gaussian_noise=0.01, *args, **kwargs):
        self.lstm_hiddens = lstm_hiddens
        self.dropout_rate = dropout_rate
        self.lstm_activation = lstm_activation
        self.conv_activation = conv_activation
        self.gaussian_noise = gaussian_noise
        self.n_series = n_series
        super().__init__(*args, **kwargs)
        
    def forecaster_architecture(self):
        input_model = tf.keras.Input(shape=(self.window_size, self.n_series))
        x = tf.keras.layers.GaussianNoise(self.gaussian_noise)(input_model)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_hiddens, activation=self.lstm_activation, use_bias=True))(x)
        x = tf.keras.layers.RepeatVector(14)(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_hiddens, activation=self.lstm_activation, return_sequences=False, use_bias=True))(x)
        x = tf.keras.layers.RepeatVector(self.window_size)(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        x = tf.keras.layers.Concatenate(axis=2)([input_model, x])
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_hiddens, activation=self.lstm_activation, return_sequences=False, use_bias=True))(x)
        x = tf.keras.layers.RepeatVector(self.steps_ahead)(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, activation=self.lstm_activation, return_sequences=True, use_bias=True))(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        x = tf.keras.layers.Conv1D(64, 1, 1, activation=self.conv_activation, use_bias=True)(x)
        output = tf.keras.layers.Conv1D(self.n_series, 1, 1, activation=self.conv_activation, use_bias=True)(x)
        
        return tf.keras.Model(input_model, output)
        
