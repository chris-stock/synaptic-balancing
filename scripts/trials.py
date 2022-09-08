"""
Code for generating trials
"""

import balancing
import pickle as pkl

### TRIAL PARAMS
def generate_task_params(
        input_noise=0.1,
        context_noise=0.1,
        num_timesteps=50,
):
    context_params = {
        'noise': context_noise,
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
        'noise': input_noise,
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
        'num_timesteps': num_timesteps,
        'context_params': context_params,
        'input_params': input_params
    }

    return task_params

def generate_trials(
    trial_data_path,
    n_trials,
    task_params
):

    # generate trials from task
    task = balancing.tasks.IntegrationTask(**task_params)
    trials = [
        task.generate_all_conditions(return_labels=True)
        for _ in range(n_trials)
    ]

    # write trials to file
    with open(trial_data_path, 'wb') as f:
        pkl.dump(trials, f, -1)

    return task
