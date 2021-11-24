import numpy as np
import pandas as pd

class Dataset():
    """
    Base class for time series datasets
    """
    def __init__(self):
        pass
    
    def load(self):
        """
        Load dataset
        must be implemented in subclass
        """
        raise NotImplementedError("you must provide an implementation for load method")


class ETSDataset(Dataset):
    """
    ETS dataset
    Sampling synthetic data from the ETS model
    """
    def __init__(self, ets_families, length=100, freq=7, positive=True):
        """
        ets_families: list of ETS families to sample from and number of samples
        """
        super().__init__()
        
        self.ets_families = ets_families
        # generating series
        self.series = []
        for family, n_series in ets_families.items():
            for i in range(n_series):
                error, trend, seasonality = family.replace(' ', '').split(',')
                if trend == 'Ad':
                    phi = np.random.uniform(low=0.95, high=1.0)
                else:
                    phi = 1.0

                series = self.generate_series(length=length, alpha=np.random.uniform(),
                                beta=np.random.uniform(),
                                phi=phi,
                                gamma=np.random.uniform(),
                                l0 = np.random.normal(),
                                b0 = np.random.normal(),
                                seasonality_freq=freq,
                                s_init=np.random.normal(size=(freq)),
                                error=error, trend=trend, seasonality=seasonality)
                
                if positive:
                    series += max(0, -series.min())
                self.series.append(series)

    def load(self, return_family=False):
        """
        Load dataset
        """
        series = np.array(self.series)
        if return_family:
            families = []
            for family, n_series in self.ets_families.items():
                for i in range(n_series):
                    families.append(family)
            return series, families
        return series

    def generate_series(self, error='A', trend='A', seasonality='A', length=100, alpha=0.6, beta=0.6, gamma=0.5, phi=1.0,
               l0=0.2, b0=0.1, error_mu=0.0, error_sigma=0.01, seasonality_freq=7, s_init=np.random.normal(size=7)):
        """
        samples a time series from ETS class
        input parameters:
        - length: length of the time series
        - error : type of the error compoenent
                'A' for additive error
                'M' for multiplicative error
        - trend : type of the trend component
                'N' for no trend
                'A' for additive trend
                'Ad' for additive damped trend
        - seasonality: type of the seasonality component
                'N' for no seasonality
                'A' for additive seasonality
                'M' for multiplicativeseasonality
        - seasonality_freq : frequency of the seasonality
        - alpha, beta, gamma, phi, l0, b0, error_mu, error_sigma : parameters of the ETS model
        """
        series = []
    
        y = []
        l = [l0]
        b = [b0]
        s = list(s_init)
        
        m = seasonality_freq
        
        if trend == 'N':
            phi = 0
        
        for i in range(length):
            et = np.random.normal(error_mu, error_sigma)
            if seasonality == 'N':
                sm = 0
            else:
                sm = s[-m]


            # additive seasonality
            if seasonality=='A' or seasonality=='N':
                st = sm
                bt = phi*b[-1]
                lt = l[-1] + phi*b[-1]
                yt = l[-1] + phi*b[-1] + sm

                # additive error
                if error == 'A':
                    st += gamma*et
                    lt += alpha*et
                    bt += beta*et
                    yt += et
                # multiplicative error
                else:
                    e_temp = l[-1] + b[-1] + sm
                    st += gamma*e_temp*et
                    bt += beta*e_temp*et
                    lt += alpha*e_temp*et
                    yt *= (1+et)

            # multiplicative seasonality
            elif seasonality=='M':
                st = sm
                bt = phi*b[-1]
                lt = l[-1] + phi*b[-1]
                yt = (l[-1] + phi*b[-1])*sm

                # additive error
                if error == 'A':
                    st += gamma*et/(l[-1] + phi*b[-1])
                    lt += alpha*et/sm
                    bt += beta*et/sm
                    yt += et
                # multiplicative error
                else:
                    st *= 1 + gamma*et
                    bt += beta*(l[-1] + phi*b[-1])*et
                    lt *= 1 + alpha*et
                    yt *= 1 + et

            y.append(yt)
            l.append(lt)
            b.append(bt)
            s.append(st)
        
        return np.array(y)
