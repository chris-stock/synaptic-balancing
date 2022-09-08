import tensorflow as tf
import numpy as np
from ..components import rmse_loss
from ..utils import draw_Wrec, draw_cov

__all__ = ['RecurrentLinearTeacherTask']


class RecurrentLinearTeacherTask(object):

    def __init__(self,
                 num_timesteps,
                 num_outputs,
                 num_neurons,
                 output_noise = 0,
                 U_init = ('I',),
                 W_in = None,
                 W_rec = None,
                 W_out = None,
                 ):

        self.num_neurons = num_neurons
        self.num_inputs = 1
        self.num_timesteps = num_timesteps
        self.num_outputs = num_outputs
        self.target_dims = [num_outputs]
        self.output_noise = output_noise

        # initialize input, recurrent, and output matrices
        if W_in is None:
            self.W_in = np.random.randn(self.num_neurons)/np.sqrt(self.num_neurons)
        else:
            self.W_in = W_in
        if W_rec is None:
            self.W_rec = np.random.randn(self.num_neurons,self.num_neurons)/np.sqrt(self.num_neurons)
        else:
            self.W_rec = W_rec
        if W_out is None:
            self.W_out = np.random.randn(self.num_neurons, self.num_outputs)/np.sqrt(self.num_neurons)
        else:
            self.W_out = W_out

        # generate input autocorrellation
        self.L = draw_cov(self.num_timesteps, U_init)

        # compute effective linear map computed by the RNN
        B = [self.W_in]
        for _ in range(self.num_timesteps-1):
            B.append(B[-1].dot(self.W_rec))
        B.reverse()
        self.B = np.array(B) # num_timesteps x num_neurons
        self.W_eff = self.B.dot(self.W_out) # num_timesteps x num_targets

    def generate_batch(self, batch_size):
        inputs = np.random.randn(batch_size, self.num_timesteps).dot(self.L.T)
        outputs = inputs.dot(self.W_eff) # batch_size x num_targets
        noise = self.output_noise * np.random.randn(*outputs.shape)
        inputs = inputs[:, :, None] # batch_size x num_timesteps x 1
        return (inputs, outputs + noise)

    def loss_function(self, outputs, target):
        """ Root-Mean-Square-Error
        """
        return rmse_loss(outputs[-1], target)

    def transform(self, outputs):
        """Transforms predictions from RNN by softmax function.
        """
        return outputs
