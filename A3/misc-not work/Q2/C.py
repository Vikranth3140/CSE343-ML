import numpy as np

class WeightInitializer:
    
    @staticmethod
    def zero_init(shape):
        """
        Initializes weights to zero.
        """
        return np.zeros(shape)
    
    @staticmethod
    def random_init(shape, scale=0.01):
        """
        Initializes weights randomly within a uniform distribution.
        """
        return np.random.uniform(-scale, scale, shape)
    
    @staticmethod
    def normal_init(shape, scale=1.0):
        """
        Initializes weights using a normal distribution with mean 0 and standard deviation scale.
        """
        return np.random.normal(0, scale, shape)