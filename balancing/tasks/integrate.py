import tensorflow as tf
import numpy as np
import itertools
from ..utils import softmax
from ..components import softmax_loss, rmse_loss

__all__ = ['IntegrationTask']

_default_context_params = {
    'noise': 0,
    'signal': 1,
    'num': 2
}

_default_input_params = {
    'noise': .1,
    'signal': 1,
    'num': 2
}


class IntegrationTask(object):
    """
    Context-Dependent Integration (Mante-Susillo)
    """

    def __init__(self,
                 num_timesteps=100,
                 context_params=_default_context_params,
                 input_params=_default_input_params
                 ):

        # task attributes we may wish to query later
        self.n_contexts = context_params['num']
        self.inputs_per_context = input_params['num']
        self.num_timesteps = num_timesteps
        self.timebase = np.arange(num_timesteps)
        self.total_inputs = self.n_contexts * self.inputs_per_context
        self.context_params = context_params
        self.input_params = input_params

    def generate_trial(self, context=None, inputs=None):
        """Generates a set of input signals and target output.

        Returns:
            context_signals, ndarray: (n_contexts x time) time series of the context signals
            input_signals, ndarray: (n_contexts * inputs_per_context x time) time series of the input signals
            target, ndarray: (inputs_per_context,) one-hot encoding of the desired network output
        """
        if context is None:
            context = np.random.randint(self.n_contexts)
        else:
            assert context >= 0 and context < self.n_contexts

        if inputs is None:
            inputs = np.random.randint(self.inputs_per_context, size=self.n_contexts)
        else:
            assert np.iterable(inputs) and len(inputs) == self.n_contexts
            for i in inputs:
                assert i >= 0 and i < self.inputs_per_context

        # context signals
        context_signals = np.random.randn(self.n_contexts,
                                          self.num_timesteps).astype(np.float32)
        context_signals *= self.context_params['noise']
        context_signals[context] += self.context_params['signal']

        # input signals
        input_signals = np.random.randn(self.n_contexts,
                                        self.inputs_per_context,
                                        self.num_timesteps).astype(np.float32)
        input_signals *= self.input_params['noise']
        input_signals[np.arange(self.n_contexts), inputs] += self.input_params['signal']

        # target network output
        target = np.squeeze(
            np.cumsum(input_signals[context], axis=1, dtype=np.float32)
        )

        # reshape input signals (total_inputs x time)
        input_signals = input_signals.reshape(-1, self.num_timesteps)

        return context_signals, input_signals, target

    def generate_batch(self, batch_size):
        inputs, targets = [], []
        for k in range(batch_size):
            c, i, t = self.generate_trial()
            inputs.append(np.vstack((c, i)).T)  # stack context & input signals
            targets.append(t)
        inputs = np.array(inputs)  # batch_size x timepoints x total_inputs
        targets = np.array(targets)  # batch_size x inputs_per_context
        return inputs, targets

    def generate_random_conditions(self, return_labels=False):
        inputs, targets = [], []
        C = range(self.n_contexts)
        I = [range(self.inputs_per_context) for _ in range(self.n_contexts)]
        clabels, ilabels = [], []
        for ci, (*ii) in itertools.product(C, *I):
            c, i, t = self.generate_trial(context=ci, inputs=ii)
            inputs.append(np.vstack((c, i)).T)
            targets.append(t)
            clabels.append(ci)
            ilabels.append(ii)
        inputs = np.array(inputs) # num_conditions x timepoints x total_inputs
        targets = np.array(targets)  # num_conditions x inputs_per_context
        if return_labels:
            return inputs, targets, clabels, ilabels
        else:
            return inputs, targets

    def generate_all_conditions(self, return_labels=False):
        inputs, targets = [], []
        C = range(self.n_contexts)
        I = [range(self.inputs_per_context) for _ in range(self.n_contexts)]
        clabels, ilabels = [], []
        for ci, (*ii) in itertools.product(C, *I):
            c, i, t = self.generate_trial(context=ci, inputs=ii)
            inputs.append(np.vstack((c, i)).T)
            targets.append(t)
            clabels.append(ci)
            ilabels.append(ii)
        inputs = np.array(inputs) # num_conditions x timepoints x total_inputs
        targets = np.array(targets)  # num_conditions x inputs_per_context
        if return_labels:
            return inputs, targets, clabels, ilabels
        else:
            return inputs, targets

    def loss_function(self, outputs, target):
        return rmse_loss(outputs, target)

    def transform(self, outputs):
        return outputs

    @property
    def num_inputs(self):
        return self.n_contexts + self.n_contexts * self.inputs_per_context

    @property
    def num_outputs(self):
        return self.inputs_per_context

    @property
    def target_dims(self):
        return [self.inputs_per_context, self.num_timesteps]