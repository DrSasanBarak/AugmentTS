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


class Sampling(layers.Layer):
    """Reparameterization trick requierd for training VAE.
       Currently supports only normal distribution, and it will be improved
    """

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAEBase(keras.Model):
    def __init__(self, **kwargs):
        super(VAEBase, self).__init__(**kwargs)
        self.encoder = self.construct_encoder()
        self.decoder = self.construct_decoder()
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(data, reconstruction), axis=(0, 1)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = 0.0001 * tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


class LSTMVAE(VAEBase):
    def __init__(self, latent_dim=8, encoder_hiddens=[256, 128, 64], decoder_hiddens=[64, 128, 256], series_len=None):
        self.latent_dim = latent_dim
        self.encoder_hiddens = encoder_hiddens
        self.decoder_hiddens = decoder_hiddens
        self.series_len = series_len
        super(LSTMVAE, self).__init__()
        
    def construct_encoder(self):
        encoder_inputs = keras.Input(shape=(1, self.series_len))
        x = keras.layers.LSTM(self.encoder_hiddens[0], return_sequences=True)(encoder_inputs)
        # stacking LSTM layers
        for i in range(1, len(self.encoder_hiddens)-1):
            x = keras.layers.LSTM(self.encoder_hiddens[i], return_sequences=True)(x)
        x = keras.layers.LSTM(self.encoder_hiddens[-1])(x)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        return encoder

    def construct_decoder(self):
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = keras.layers.RepeatVector(1)(latent_inputs)
        # stacking LSTM layers
        for h in self.decoder_hiddens:
            x = keras.layers.LSTM(h, return_sequences=True)(x)
        
        decoder_outputs = keras.layers.TimeDistributed(keras.layers.Dense(self.series_len))(x)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        return decoder

class VAEAugmenter():
    """
    A class for augmenters that use variational autoencoder
    """
    def __init__(self, vae):
        self.vae = vae
        # compiling the vae model
        self.vae.compile(optimizer=keras.optimizers.Adam())

        self.fit = self.vae.fit
        self.latent_dim = vae.latent_dim

    def sample(self, n=1, X=np.array([]), sigma=0.1):
        """
        Generating new time series

        Parameters
        ----------
        n : the number of generated series, ignored when X is not empty
        X : if it set to None then the samples will be from the whole latent space
            if it is a dataset then the sample will be simillar to the provided data
        sigma : the standard deviation of noise that affect the variety of generated samples
        """
        if X.size > 0:
            # encode the data
            z = self.vae.encoder(X)[2]
            # adding gaussian noise to the latent values
            noise = sigma * tf.keras.backend.random_normal(shape=(X.shape[0], self.latent_dim))
            z += noise
            # reconstruct the time series
            return self.vae.decoder(z)
        else:
            # generate random points in the latent space
            z = sigma * tf.keras.backend.random_normal(shape=(n, self.latent_dim))
            # decode the latent values
            return self.vae.decoder(z)

    def latent(self, X):
        """
        Generating latent codes for a dataset

        Parameters
        ----------
        X : dataset
        """
        # encode the data
        return self.vae.encoder(X)[2]
