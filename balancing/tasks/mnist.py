import numpy as np
from ..components import softmax_loss
from ..utils import no_noise
import functools
from tensorflow.examples.tutorials.mnist import input_data

__all__ = ['MNIST']

class MNIST(object):

    def __init__(self,
                 noise = no_noise,
                 sampling_ratios = np.full(10, .1),
                 permute_images = False,
                 loss = softmax_loss
                 ):

        # load data
        self._data = input_data.read_data_sets('MNIST_data', one_hot=True)
        
        # get training and testing data
        self.train_images = self._data.train.images
        self.train_labels = self._data.train.labels
        self.test_images = self._data.test.images
        self.test_labels = self._data.test.labels

        # indices for each digit
        self.train_lookup = [np.argwhere(self.train_labels[:,i]).ravel() for i in range(10)]
        self.test_lookup = [np.argwhere(self.test_labels[:,i]).ravel() for i in range(10)]

        # sampling ratios (to test imbalanced data)
        if len(sampling_ratios) != 10:
            raise ValueError('Need to specify sampling ratios for all ten digits.')
        self.sampling_ratios = np.array(sampling_ratios)
        if np.any(self.sampling_ratios < 0):
            raise ValueError('Sampling ratios must all be nonnegative.')
        self.sampling_ratios = self.sampling_ratios / np.sum(self.sampling_ratios)
        
        # input noise
        self.noise = noise
        self._loss = loss
        # self._transform = transform

        # if desired, permute images
        if permute_images:
            # different permutation for each digit
            for i, j in zip(self.train_lookup, self.test_lookup):
                k = np.random.permutation(28**2)
                self.train_images[i] = self.train_images[i][:,k]
                self.test_images[j] = self.test_images[j][:,k]

    def generate_batch(self, batch_size, mode='train', pvals=None):
        
        # fetch data
        if mode == 'train':
            xi = self.train_lookup
            x = self.train_images
            y = self.train_labels
        elif mode == 'test':
            xi = self.test_lookup
            x = self.test_images
            y = self.test_labels
        else:
            raise ValueError('mode not understood')

        # how many images for each digit in this batch
        pvals = self.sampling_ratios if pvals is None else pvals
        N = np.random.multinomial(batch_size, pvals=pvals)
        
        # sample images for each digit
        images = np.concatenate([np.random.choice(xi[i], size=n) for i, n in enumerate(N)])

        # add noise and return
        return (self.noise(x[images]), y[images])

    def loss_function(self, outputs, target):
        return self._loss(outputs, target)

    def transform(self, outputs):
        """Transforms predictions from network by softmax function.
        """
        # return self._transform(outputs, axis=-1)
        raise AssertionError('Not implemented...')

    @property
    def num_inputs(self):
        return 28*28

    @property
    def num_outputs(self):
        return 10

