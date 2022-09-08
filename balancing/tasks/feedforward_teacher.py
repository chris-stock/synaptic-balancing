import numpy as np
from ..components import rmse_loss, relu
from ..utils import draw_cov
from ..feedforward import Feedforward

__all__ = ['FeedforwardTeacherTask']


class FeedforwardTeacherTask(object):

    def __init__(self,
                 layer_sizes,
                 nonlinearity = relu,
                 output_noise = 0,
                 X_cov = ('I',),
                 **net_kw
                 ):


        self.num_inputs = layer_sizes[0]
        self.num_outputs = layer_sizes[-1]
        self.num_layers = len(layer_sizes)-1
        self.nonlinearity = nonlinearity

        self.net = Feedforward(self, layer_sizes,
            nonlinearity=self.nonlinearity, **net_kw)
        self.weights = self.net.sess.run(self.net.weights)

        # generate correlated (Wishart distributed) inputs
        self.L = draw_cov(self.num_inputs, X_cov)
        self.output_noise = output_noise

    def generate_batch(self, batch_size):
        x = np.random.randn(batch_size, self.num_inputs).dot(self.L.T)
        y = self.net.transform(x)[-1]
        noise = self.output_noise * np.random.randn(*y.shape)
        return (x, y + noise)

    def loss_function(self, outputs, target):
        """ Root-Mean-Square-Error
        """
        return rmse_loss(outputs, target)
