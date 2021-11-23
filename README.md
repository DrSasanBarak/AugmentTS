# AugmentTS :: Time Series Data Augmentation using Deep Generative Models
**Note!!!** The package is under development so be careful for using in production!
## Features
- Time Series Data Augmentation using Deep Generative Models
- Visualizing the Latent Space of Generative Models
- Time Series Forecasting using Deep Neural Networks

## Installation
You can install the last stable version using pip
```
pip install augmentts
```
## How to Use
### Augmentation Guide
#### Create an augmenter
```python
from augmentts.augmenters.vae import LSTMVAE, VAEAugmenter

# create a variational autoencoder
vae = LSTMVAE(series_len=100)
# create an augmenter
augmenter = VAEAugmenter(vae)
```
The above code uses the default settings for the LSTM-VAE model. You can customize its architecture or use your own model for encoder and decoder. Note currently we only support Keras models.  
#### Train the augmenter
```python
augmenter.fit(data, epochs=50, batch_size=32)
```
#### Generate new time series!
Two strategies for sampling have been implemented.  
You can simply sample from the latent space. Here `n` is the number of generated series
```python
augmenter.sample(n=1000)
```
You also can generate time series by reconstructing a set of series.
```python
augmenter.sample(X=data)
```
In both cases you can control the variety of generated time series using `sigma`
```python
augmenter.sample(n=1000, sigma=0.2)
```
### Forecasting Guide
First we create a random dataset then use `prepare_ts` helper function to prepare the dataset for forecasting.
```python
from augmentts.forecasters.deep import LSTMCNNForecaster
from augmentts.utils import prepare_ts
import numpy as np

# creating a random dataset
ts = np.random.rand(100, 10)
# preparing data for rolling window regression
X, y = prepare_ts(ts, 8, 4)
```
Now we can create a forecaster and train it. Note the `fit` function is just an alias for Keras fit function thus you can pass all of the supported arguments of Keras fit function.
```python
model = LSTMCNNForecaster(window_size=8, steps_ahead=4, n_series=10)
model.fit(X, y, epochs=10)
```
After training you can use `predict` to evaluate the model. 
```python
model.predict(X)
```


## Supported Augmenters
Supported models for augmentation currently are as follows:
|  Model  |           Type          |   Supported Time Series  |                                Description                                |
|:-------:|:-----------------------:|:------------------------:|:-------------------------------------------------------------------------:|
| LSTMVAE | Variational Autoencoder | Univariate, fixed length | A Variational Autoencoder with stacked LSTM layers for encoder and decoder based on the paper [paper citation] |

## Supported Forecasters
Currently an LSTM-CNN forecaster is implemented. You can either customize it or just implement your own architecture.

## Examples
### Augmenting ETS Time Series 
Let's see how to use AugmentTS to generate time series similiar to one of the ETS families.  
```python
import matplotlib.pyplot as plt
import seaborn as sb
sb.set(style='white')
import pandas as pd
```
Using `ETSDataset` class we can sample time series from any ETS model.  
```python
from augmentts.datasets import ETSDataset
```
For the sake of simplicity we sample 60 series from ANA model (Additive error, No trend, Additive seasonality) and 30 seris from MNN model (Multiplicative error, no trend, no seasonality):
```python
# sampling a few series from ETS model
ets = ETSDataset(ets_families={
    'A,N,A' : 60, # 60 samples from ANA model
    'M,N,N' : 30  # 30 samples from MNN model
}, length=100)

ts_data, family = ets.load(return_family=True)
```
We can use any dimensionality reduction or manifold learning method for visulizing the series in plane. Let's just use t-SNE.
```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2)
z = tsne.fit_transform(ts_data)
```
We simply use Pandas and Seaborn to draw a scatte plot
```python
original_df = pd.DataFrame({'family' : family})
original_df[['z1', 'z2']] = z

sb.scatterplot(data=original_df, x='z1', y='z2', hue='family')
```
![image](https://user-images.githubusercontent.com/8543469/143130228-28473bcd-1201-403e-ba73-76b390609839.png)

Now we use AugmentTS to augment the MNN family:
```python
from augmentts.augmenters.vae import LSTMVAE, VAEAugmenter

# creating the VAE
vae = LSTMVAE(series_len=100, encoder_hiddens=[512, 256, 128], decoder_hiddens=[128, 256, 512])
augmenter = VAEAugmenter(vae)
# training the VAE on MNN family
vae_data = ts_data[-30:, :].reshape(-1, 1, 100)
augmenter.fit(vae_data, epochs=100, batch_size=16)
```
Generating 30 new time series.
```python
n_generated = 30
generated = augmenter.sample(n=n_generated, sigma=0.5)
generated = generated.numpy()[:, 0, :]
```
Now we visualize the augmented time series and the original ones
```python
z = tsne.fit_transform(np.vstack([ts_data, generated]))
augmented_df = pd.DataFrame({'family' : family + ['Generated M,N,N']*n_generated})
augmented_df[['z1', 'z2']] = z
sb.scatterplot(data=augmented_df, x='z1', y='z2', hue='family')
```
Here is the result of augmentation!  
![image](https://user-images.githubusercontent.com/8543469/143130434-57e70b76-c242-4f8d-9a0e-44659d83d3e1.png)


## Contributors
The list of the current contributors:
- Sasan Barak
- Amirabbas Asadi
- Ehsan Mirafzali
- Mohammad Joshaghani
