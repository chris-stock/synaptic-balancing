import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import balancing
from balancing import tasks, BalancingRNN, sgd_optimizer
from balancing.components import orth_initializer, randn_initializer, identity_initializer, custom_initializer
from balancing.components import relu, frob_regularizer, no_regularizer
from os.path import join
import os
import pickle as pkl
import datetime
train_results = None # will be defined and used later by rnn.train


### NETWORK & TRAINING PARAMETERS

l2_reg_scale = np.array([0., .1, .3], dtype=float)
# l2_reg_scale=np.array([0.], dtype=float)

N = 256
# N = 500
n_iter = 3000 #3000 #6000
learning_rate = 1e-2 #5e-3


### TRIAL PARAMS
context_params = {
    'noise': .3,
    'signal': 1,
    'num': 2,
    'onset': {
        'low': 0,
        'high': 1
    },
    'offset': {
        'low': 100,
        'high': 101
    }
}

input_params = {
    'noise': .3, #1
    'signal': 1,
    'num': 2,    'onset': {
        'low': 0,
        'high': 1
    },
    'offset': {
        'low': 100,
        'high': 101
    }
}

task_params = {
    'num_timesteps': 50,
    'context_params': context_params,
    'input_params': input_params
}

n_trials = 10000 # number of trials to generate of the task

####  DATA PATHS
THIS = 'N-{}_ctx-noise-{}_input-noise-{}_{}'.format(
    N,
    context_params['noise'],
    input_params['noise'],
    datetime.datetime.utcnow().isoformat()
    )
data_dir = '../data/neural-gradients-regularized-training/'
trial_fpath = join(data_dir, 'integration_trials_{}.pkl'.format(THIS))
fig_dir = '../figures_new/trained-networks'

data_dir = '../data/neural-gradients-regularized-training/'
data_fpath = os.path.join(data_dir, 'imbalances-during-regularized-training-{}.pkl'.format(THIS))

### GENERATE TRIALS
task = balancing.tasks.IntegrationTask(**task_params)
canon_inputs, canon_targets, clabels, ilabels = task.generate_all_conditions(return_labels=True)
n_conditions = len(clabels)
trials = [task.generate_all_conditions(return_labels=True) for _ in range(n_trials)]
with open(trial_fpath, 'wb') as f:
    pkl.dump(trials, f, -1)


### TRAIN NETWORKS

def train_with_balancing_regularization(N, l2_rate, initial_weights=None, learning_rate=1e-2, niter=1600):
    calc_regularizer = lambda rate: no_regularizer if rate==0 else frob_regularizer(l2_rate)
    
    # set RNN parameters
    clipping = {
        'method': tf.clip_by_global_norm,
        'args': (10.0,) #(10.0,)
    }
    p = 2
    
    if initial_weights is None:
        initializers = {
            'W_rec': randn_initializer(stddev=2./np.sqrt(N)),
            'W_in': randn_initializer(stddev=1/np.sqrt(N)),
            'W_out': randn_initializer(stddev=1/np.sqrt(N)),
            'b':  lambda s, _: np.zeros(s),                    
            }
    else:
        initializers = {k: custom_initializer([w]) for k, w in initial_weights.items()}
    
    rnn_params = {
        'p': p,
        'num_neurons': N,
        'nonlinearity': relu,
        'balancing_penalty_strength': l2_rate,    
        'initializers': initializers
    }

    # make RNN
    rnn = BalancingRNN(task, **rnn_params)
    initial_weights = rnn.dump_weights()
    train_op = sgd_optimizer(rnn, clipping=clipping)
    train_results = None

    # set training parameters
    train_args = {
        'train_op': train_op,
        'niter': niter,
        'learning_rate': learning_rate,
        'append_to': train_results,
        'trial_generator': task.generate_all_conditions,
        'save_weights': True,
    }

    # train network
    train_results = rnn.train(**train_args)

    return rnn, train_results

initial_weights = {
    'W_rec': randn_initializer(stddev=1./np.sqrt(N))((N,N), 0),
    'W_in': randn_initializer(stddev=1./np.sqrt(N))((task.num_inputs,N), 0),
    'W_out': randn_initializer(stddev=1./np.sqrt(N))((N,task.num_outputs), 0),
    'b': np.zeros((N,))
}    

train_args = [{
    'N': N,
    'l2_rate': l2_reg,
    'niter': n_iter,
    'initial_weights': initial_weights,
    'learning_rate': learning_rate,
} for l2_reg in l2_reg_scale]

results = [train_with_balancing_regularization(**args) for args in train_args]
rnns, train_results = zip(*results)

### SAVE DATA
save_data = {
    'g_norm': [
        np.linalg.norm(res['balancing_imbalances'], axis=1) for res in train_results
    ],
    'loss': [
        res['loss'] for res in train_results
    ],
    'balancing_cost': [
        res['balancing_cost'] for res in train_results
    ],
    'initial_final_weights': [
        (res['weights'][0], res['weights'][-1]) for res in train_results
    ],
    'l2_reg_scale': l2_reg_scale,
}

with open(data_fpath, 'wb') as f:
    pkl.dump(save_data, f, -1)

