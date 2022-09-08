import numpy as np
import random


class FiniteDataset(object):
    """Wraps any task object to accomodate finite-size data"""
    
    def __init__(self, task, size):
        
        # expose information from underlying task
        self.subtask = task
        self.num_inputs = self.subtask.num_inputs
        self.num_outputs = self.subtask.num_outputs
        self.loss_function = self.subtask.loss_function

        # generate data
        self.Xtrain, self.Ytrain = self.subtask.generate_batch(size)

        # iteration order for minibatch training
        self._idx = []

        # keep track of number of epochs
        self.epoch_count = -1

    def generate_batch(self, batch_size):
        
        # sample inputs
        idx = []
        while len(idx) < batch_size:

            # sample another image
            try:
                idx.append(self._idx.pop())

            # new epoch
            except IndexError:
                self._idx = list(range(len(self.Xtrain)))
                random.shuffle(self._idx)
                self.epoch_count += 1

        return self.Xtrain[idx], self.Ytrain[idx]
