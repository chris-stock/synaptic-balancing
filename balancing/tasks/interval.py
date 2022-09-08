import tensorflow as tf
import numpy as np
import itertools
from ..components import rmse_loss
from scipy.ndimage import gaussian_filter1d

__all__ = ['IntervalProduction']

class IntervalProduction(object):

    def __init__(self,
                 input_noise = 0.1,
                 sample_go_onset = lambda : np.random.randint(5, 21),
                 sample_go_duration = lambda : 5,
                 intervals = [25, 75],
                 num_timesteps = 100,
                 target_width = 1
                 ):

        # task attributes we may wish to query later
        self.input_noise = input_noise
        self.num_timesteps = num_timesteps
        self.timebase = np.arange(num_timesteps)
        self.total_inputs = len(intervals)
        self.sample_go_onset = sample_go_onset
        self.sample_go_duration = sample_go_duration
        self.intervals = np.array(intervals)
        self.target_width = target_width
        
    def generate_trial(self, interval=None):
        """Generates a set of input signals and target output.

        Returns:
            inputs, ndarray: (num_inputs x num_timesteps) time series of the input signals
            target, ndarray: (num_timesteps,) one-hot encoding of the desired network output
        """

        # create noisy input signals
        inputs = self.input_noise * np.random.randn(self.num_inputs, self.num_timesteps)

        # determine which input is active
        if interval is None:
            i = np.random.randint(0, self.num_inputs)
            interval = self.intervals[i]
        elif interval not in self.intervals:
            raise ValueError(('Specified interval {} not found. ' + 
                             'Choose from {}, or create a new task.'.format(interval, self.interval)))
        else:
            i = np.where(self.intervals == interval)[0][0]

        # add step function to active input
        t0 = self.sample_go_onset()          # variable onset
        t1 = t0 + self.sample_go_duration()  # variable duration
        print(t0, t1)
        inputs[i, t0:t1] += 1

        # generate target
        target = np.zeros(self.num_timesteps)
        target[t1 + interval] = 1
        target = gaussian_filter1d(target, self.target_width)
        
        return inputs, target

    def generate_batch(self, batch_size):
        inputs, targets = [], []
        for k in range(batch_size):
            i, t = self.generate_trial()
            inputs.append(i)
            targets.append(t)
        inputs = np.array(inputs) # batch_size x timepoints x total_inputs
        targets = np.array(targets)  # batch_size x inputs_per_context
        return inputs, targets

    def generate_all_conditions(self):
        inputs, targets = [], []
        for interval in self.intervals:
            i, t = self.generate_trial(interval=interval)
            inputs.append(i)
            targets.append(t)
        inputs = np.array(inputs) # batch_size x timepoints x total_inputs
        targets = np.array(targets)  # batch_size x inputs_per_context
        return inputs, targets

    def loss_function(self, outputs, target):
        losses = tf.map_fn(lambda o: rmse_loss(o, target), tf.stack(outputs))
        return tf.reduce_mean(losses, axis=0)

    def transform(self, outputs):
        """No transformation on outputs
        """
        return tf.identity(outputs)

    @property
    def num_inputs(self):
        return len(self.intervals)

    @property
    def num_outputs(self):
        return 1

