import numpy as np
from ..components import softmax_loss

__all__ = ['SphereClassification']


class SphereClassification(object):

    def __init__(self,
                 num_inputs,  # dimensionality of input space
                 norms,  # expected L2 norm of each category
                 noise=0,  # amount of noise added to each example
                 category_ratios=None,
                 ):

        self.num_inputs = num_inputs
        self.norms = norms
        self.noise = noise
        self.category_ratios = category_ratios

    def generate_batch(self, batch_size):
        # cluster ids
        ids = np.random.choice(len(self.norms), size=batch_size, p=self.category_ratios)

        # inputs
        x = np.random.randn(batch_size, self.num_inputs)
        x *= self.norms[ids, None] / np.linalg.norm(x, axis=-1, keepdims=True)
        x_noise = self.noise * np.random.randn(batch_size, self.num_inputs)

        # labels
        y = np.zeros((batch_size, len(self.norms)))
        y[np.arange(batch_size), ids] = 1.0

        return (x + x_noise, y)

    def loss_function(self, outputs, target):
        return softmax_loss(outputs, target)

    @property
    def num_outputs(self):
        return len(self.norms)