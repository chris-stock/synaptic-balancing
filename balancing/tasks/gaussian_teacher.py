import numpy as np
from ..components import rmse_loss

__all__ = ['GaussianTeacherTask']


class GaussianTeacherTask(object):

    def __init__(self,
                 func,
                 num_inputs,
                 num_outputs,
                 num_datapoints=None,
                 loss=rmse_loss
                 ):

        self.f = func
        self.loss = rmse_loss
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_datapoints = num_datapoints

        if num_datapoints is not None:
            self._x = np.random.randn(num_datapoints, num_inputs)
            self._i = 0
        else:
            self._x = None

    def generate_batch(self, batch_size):
        if self.num_datapoints is None:
            x = np.random.randn(batch_size, self.num_inputs)
        elif batch_size > self.num_datapoints:
            raise ValueError('Batch size larger than dataset')
        else:
            x = np.roll(self._x, -self._i, axis=0)[:batch_size]
            self._i = (self._i + batch_size) % self.num_datapoints
        return (x, self.f(x))

    def loss_function(self, outputs, target):
        """ Root-Mean-Square-Error
        """
        return self.loss(outputs, target)
