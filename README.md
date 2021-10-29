# TAug :: Time Series Data Augmentation using Deep Generative Models
**Note!!!** The package is under development so be careful for using in production!
## Features
- Time Series Data Augmentation using Deep Generative Models
- Visualizing the Latent Space of Generative Models
- Time Series Forecasting using Deep Neural Networks

## Installation
You can install the last stable version using pip
```
pip install taug
```
## How to Use
### Augmentation Guide
#### Create an augmenter
```python
from taug.augmenters.vae import LSTMVAE
from taug.augmenters.vae import VAEAugmenter

# create a variational autoencoder
vae = LSTMVAE(series_len=100)
# use the created vae as an augmenter
augmenter = VAEAugmenter(vae)
```
The above code uses the default settings for the LSTM-VAE model. You can customize its architecture or use your own model for encoder and decoder. Note currently we only support Keras models.  
#### Train the augmenter
```python
augmenter.fit(data, epochs=64)
```
#### Generate new time series!
Two strategy for sampling have been implemented.  
You can simply sample from the latent space. Here `n` is the number of generated series
```python
augmenter.sample(n=1000)
```
You also can generate time series by reconstructing a set of series.
```python
augmenter.sample(X=data)
```
In latter case you can control the variety of generated time series using `sigma`
```python
augmenter.sample(X=data, sigma=0.2)
```

### Forecasting Guide
[todo] Forecasting guide will be here!

## Supported Augmenters
Supported models for augmentation currently are as follows:
|  Model  |           Type          |   Supported Time Series  |                                Description                                |
|:-------:|:-----------------------:|:------------------------:|:-------------------------------------------------------------------------:|
| LSTMVAE | Variational Autoencoder | Univariate, fixed length | A Variational Autoencoder with stacked LSTM layers for encoder and decoder based on the paper [paper citation] |

## Supported Forecasters
Supported models for time series forecasting are as follows:

## Contributors
The list of the current contributors:
- Sasan Barak
- Amirabbas Asadi
