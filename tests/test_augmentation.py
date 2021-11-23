from augmentts.augmenters.vae import LSTMVAE, VAEAugmenter
import numpy as np

data = np.random.normal(size=(40, 1, 100))

vae = LSTMVAE(series_len=100)
augmenter = VAEAugmenter(vae)
augmenter.fit(data, epochs=50, batch_size=32)

samples = augmenter.sample(n=10)
print(samples.shape)