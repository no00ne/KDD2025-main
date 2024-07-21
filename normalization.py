import numpy as np
import os



class Scaler:
    def transform(self, data):

        raise NotImplementedError("Transform not implemented")

    def inverse_transform(self, data):

        raise NotImplementedError("Inverse_transform not implemented")


class NoneScaler(Scaler):
    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


class NormalScaler(Scaler):
    """
    x = x / x.max
    """

    def __init__(self, maxx):
        self.max = maxx

    def transform(self, data):
        return data / self.max

    def inverse_transform(self, data):
        return data * self.max


class StandardScaler(Scaler):
    """
    Z-score
    x = (x - x.mean) / x.std
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class MinMax01Scaler(Scaler):
    """
    MinMax[0, 1]
    x = (x - min) / (max - min)
    """

    def __init__(self, minn, maxx):
        self.min = minn
        self.max = maxx 

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        return data * (self.max - self.min) + self.min


class MinMax11Scaler(Scaler):
    """
    MinMax[-1, 1]
    x = (x - min) / (max - min)
    x = x * 2 - 1
    """

    def __init__(self, minn, maxx):
        self.min = minn
        self.max = maxx

    def transform(self, data):
        return ((data - self.min) / (self.max - self.min)) * 2. - 1.

    def inverse_transform(self, data):
        return ((data + 1.) / 2.) * (self.max - self.min) + self.min


class LogScaler(Scaler):
    """
    Log scaler
    x = log(x+eps)
    """

    def __init__(self, eps=0.999):
        self.eps = eps

    def transform(self, data):
        return np.log(data + self.eps)

    def inverse_transform(self, data):
        return np.exp(data) - self.eps
