import numpy as np
from ..components import rmse_loss
from ..utils import draw_Wrec, draw_cov

__all__ = ['RecurrentLinearRegressionTask']


class RecurrentLinearRegressionTask(object):

    def __init__(self,
                 num_timesteps,
                 num_outputs,
                 num_neurons,
                 output_noise = 0,
                 U_init = ('I',),
                 A_init = None,
                 ):

        self.num_neurons = num_neurons
        self.num_inputs = 1
        self.num_timesteps = num_timesteps
        self.num_outputs = num_outputs
        self.target_dims = (num_outputs,)
        
        # generate input, recurrent, and output matrices
        self.Win = np.random.randn(self.num_neurons)/np.sqrt(self.num_neurons)
        self.Wrec = np.random.randn(self.num_neurons,self.num_neurons)/np.sqrt(self.num_neurons)
        self.Wout = np.random.randn(self.num_neurons, self.num_outputs)/np.sqrt(self.num_neurons)

        # generate input autocorrellation
        self.L = draw_cov(self.num_timesteps, U_spec)

        B = [self.Win]
        for _ in range(self.num_timesteps):
            B.append(B[-1].dot(self.Wrec))
        B.reverse()
        self.B = np.array(B).T
        self.Weff = B.dot(C)

        self.output_noise = output_noise

    def generate_batch(self, batch_size):
        x = np.random.randn(batch_size, self.num_timesteps).dot(self.L.T)
        y = x.dot(self.U).dot(self.Vt)
        noise = self.output_noise * np.random.randn(*y.shape)
        x = x[:, :, None] # batch_size x num_timesteps x 1
        return (x, y + noise)

    def generate_batch(self, batch_size):

        return (inputs, targets)

    def loss_function(self, outputs, target):
        """ Root-Mean-Square-Error
        """
        return rmse_loss(outputs, target)

    def transform(self, outputs):
        """Transforms predictions from RNN by softmax function.
        """
        return softmax(outputs, axis=-1)
