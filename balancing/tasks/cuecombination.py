import numpy as np
from functools import partial
from itertools import starmap

from ..components import rmse_loss
from ..utils import gaussian, multiplicative_gaussian_noise
from ..utils import additive_gaussian_noise, compose

# TODO - make this work with recurrent architectures

__all__ = ['CueCombinationTask']


class CueCombinationTask(object):
    """
    Combination of two sensory modalities
    """

    def __init__(self,
                 neurons_per_modality = 30,
                 num_modalities = 2,
                 tuning_functions = None,
                 input_noise = None
                 ):

        # task attributes
        self.num_modalities = num_modalities
        self.neurons_per_modality = neurons_per_modality

        # default tuning curves
        if tuning_functions is None:
            if num_modalities != 2:
                raise ValueError('if num_modalities is not 2, must specify tuning functions for each modality.')
            else:
                tuning_functions = [partial(gaussian, mean=0, amplitude=1, stddev=.2),
                                    partial(gaussian, mean=0, amplitude=2, stddev=.05)]
        
        # default input noise
        if input_noise is None:
            if num_modalities != 2:
                raise ValueError('if num_modalities is not 2, must specify input noise distributions for each modality.')
            else:
                input_noise = [partial(poisson_noise, stddev=1),
                               partial(poisson_noise, stddev=1)]


        print(tuning_functions)
        print(input_noise)

        # tuning curves for each neuron in each modality
        self.tuning_curves = list(starmap(compose, zip(input_noise, tuning_functions)))

    def generate_trial(self):
        """Generate a trial in a quickly interpretable format.

        Returns
        -------
        inputs, ndarray : (num_modalities x neurons_per_modality)
        targets, float : true position of stimulus, scalar between zero and one
        """

        # stimulus position
        target = np.random.rand()
        print(target)

        # response for each sensory neuron
        inputs = np.zeros((self.num_modalities, self.neurons_per_modality))
        for m, f in enumerate(self.tuning_curves):
            inputs[m] = f(np.linspace(0, 1, self.neurons_per_modality) - target)

        return np.array(inputs), target

    def generate_batch(self, batch_size):
        """Generate a batch in a standard format for training

        Returns
        -------
        inputs : (batch_size x total_inputs)
        targets : true position (1,)
        """
        inputs, targets = [], []
        for k in range(batch_size):
            i, t = self.generate_trial()
            inputs.append(i.ravel()) # all modalities in one input vector
            targets.append([t])
        inputs = np.array(inputs) # batch_size x total_inputs
        targets = np.array(targets)  # batch_size x 1
        return inputs, targets

    def loss_function(self, output, target):
        return rmse_loss(output, target)

    def transform(self, outputs):
        """Transforms predictions from network, e.g. by sigmoid or softmax. In this case, no transform.
        """
        return outputs

    @property
    def num_inputs(self):
        return self.neurons_per_modality * self.num_modalities

    @property
    def num_outputs(self):
        return 1

    @property
    def target_dims(self):
        return [1]
