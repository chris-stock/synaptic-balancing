import numpy as np
from ..components import rmse_loss
from ..utils import draw_cov

__all__ = ['LinearRegressionTask']


class LinearRegressionTask(object):

    def __init__(self,
                 num_inputs,
                 num_outputs,
                 output_noise = 0,
                 singular_values = [1000, 100, 10],
                 X_cov = ('I',)
                 ):

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.target_dims = (num_outputs,)
        self.rank = len(singular_values)

        # generate correlated (Wishart distributed) inputs
        self.L = draw_cov(self.num_inputs, X_cov)
        sqrt_s = np.sqrt(singular_values)

        # random orthogonal matrices
        U = np.linalg.qr(np.random.randn(num_inputs, self.rank))[0]
        V = np.linalg.qr(np.random.randn(num_outputs, self.rank))[0]

        self.U = U * sqrt_s
        self.Vt = sqrt_s[:,None] * V.T

        self.output_noise = output_noise

    def generate_batch(self, batch_size):
        x = np.random.randn(batch_size, self.num_inputs).dot(self.L.T)
        y = x.dot(self.U).dot(self.Vt)
        noise = self.output_noise * np.random.randn(*y.shape)
        return (x, y + noise)

    def loss_function(self, outputs, target):
        """ Root-Mean-Square-Error
        """
        return rmse_loss(outputs, target)

    def transform(self, outputs):
        """Transforms predictions from RNN by softmax function.
        """
        return softmax(outputs, axis=-1)
