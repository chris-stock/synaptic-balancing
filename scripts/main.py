"""
Main routine
"""

import numpy as np
from os.path import join
import os
import datetime

from sbplos.trials import generate_task_params, generate_trials
from sbplos.training import train

### NETWORK & TRAINING PARAMETERS

# task parameters
input_noise = 0.3
context_noise = 0.3
num_timesteps = 50

# network parameters
N = 256

# training parameters
l2_reg_scale = np.array([0., .1, .3], dtype=float)
n_iter = 6000
learning_rate = 5e-3

# evaluation parameters
n_trials = 1000 # number of trials to generate of the task

### DIRECTORY STRUCTURE

run_id = 'N-{}_context-noise-{}_input-noise-{}_{}'.format(
    N,
    context_noise,
    input_noise,
    datetime.datetime.utcnow().isoformat()
    )

scratch_dir = os.path.expanduser('~/scratch')
run_dir = join(
    scratch_dir,
    'synaptic-balancing-PLOS',
    'regularized_network_training',
    run_id
)
os.makedirs(run_dir, exist_ok=True)

data_dir = join(run_dir, 'data')
figures_dir = join(run_dir, 'figures')
os.mkdir(data_dir)
os.mkdir(figures_dir)

trial_data_path = join(data_dir, 'trials.pkl')
trained_network_data_path = join(data_dir, 'trained_network.pkl')


### GENERATE TASK & TRIALS
task_params = generate_task_params(input_noise, context_noise)
task = generate_trials(
    trial_data_path,
    n_trials,
    task_params
)


### TRAIN NETWORKS
train_results = train(
    trained_network_data_path,
    task,
    N,
    n_iter,
    learning_rate,
    l2_reg_scale
)