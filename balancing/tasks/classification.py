import numpy as np
from ..components import softmax_loss

__all__ = ['InfiniteGaussianMixtureTask', 'FixedGaussianMixtureTask']


class InfiniteGaussianMixtureTask(object):

    def __init__(self,
                 num_inputs, # dimensionality of input space
                 num_clusters, # number of categories
                 noise, # amount of noise around each cluster
                 cluster_ratios=None # prevalance of each cluster (probability vector)
                 ):

        self.num_clusters = num_clusters
        self.num_inputs = num_inputs
        self.cluster_centers = np.random.randn(self.num_clusters, self.num_inputs)
        self.cluster_ratios = cluster_ratios
        self.num_outputs = num_clusters

        if not np.iterable(noise):
            self.noise = np.full((num_clusters, num_inputs), noise)
        else:
            self.noise = noise


    def generate_batch(self, batch_size):        
        # cluster ids
        ids = np.random.choice(self.num_clusters, size=batch_size, p=self.cluster_ratios)
        
        # inputs
        x = self.cluster_centers[ids, :]
        x_noise = self.noise[ids, :] * np.random.randn(batch_size, self.num_inputs)
        
        # labels
        y = np.zeros((batch_size, self.num_clusters))
        y[np.arange(batch_size), ids] = 1.0
        
        return (x + x_noise, y)

    def loss_function(self, outputs, target):
        return softmax_loss(outputs, target)


class FixedGaussianMixtureTask(object):

    def __init__(self,
                 num_inputs, # dimensionality of input space
                 num_clusters, # number of categories
                 noise, # amount of noise around each cluster
                 train_size, # amount of noise around each cluster
                 test_size, # amount of noise around each cluster
                 cluster_ratios=None # prevalance of each cluster (probability vector)
                 ):

        self.num_clusters = num_clusters
        self.num_inputs = num_inputs
        self.cluster_centers = np.random.randn(self.num_clusters, self.num_inputs)
        self.cluster_ratios = cluster_ratios
        self.num_outputs = num_clusters

        if not np.iterable(noise):
            self.noise = np.full((num_clusters, num_inputs), noise)
        else:
            self.noise = noise

        self.train_size = train_size
        self.test_size = test_size

        self.training_data = self._gen_data(train_size)
        self.test_data = self._gen_data(test_size)

    def generate_batch(self, batch_size, mode='train'):
        
        if mode == 'train':
            sz = self.train_size
            X, Y = self.training_data
        elif mode == 'test':
            sz = self.test_size
            X, Y = self.test_data
        else:
            raise ValueError('Mode not understood')

        i = np.random.choice(sz, size=batch_size)
        return X[i], Y[i]


    def _gen_data(self, size):        
        # cluster ids
        ids = np.random.choice(self.num_clusters, size=size, p=self.cluster_ratios)
        
        # inputs
        x = self.cluster_centers[ids, :]
        x_noise = self.noise[ids, :] * np.random.randn(size, self.num_inputs)
        
        # labels
        y = np.zeros((size, self.num_clusters))
        y[np.arange(size), ids] = 1.0
        
        return (x + x_noise, y)

    def loss_function(self, outputs, target):
        return softmax_loss(outputs, target)
