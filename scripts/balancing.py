
import rnnops
import pickle as pkl

from os.path import join
import os
import numpy as np
from scipy.ndimage import gaussian_filter1d

from rnnops.ops.balancing import robustness_cost_fn
from rnnops.ops.balancing import solve_balancing
from rnnops.trial import trial_apply

def run_balancing(
    data_dir,
    network_data_fname='trained_network.pkl',
    trial_data_fname='trials.pkl',
    output_data_fname='balanced_network.pkl',
    num_train_trials=300,
    num_eval_trials=300,
    dt=0.1,
    noise_levels=np.arange(0.4, step=0.5),
    balancing_T_max=3000
):

    ### LOAD DATA
    with open(join(data_dir, network_data_fname), 'rb') as f:
        weight_data = pkl.load(f)
    with open(join(data_dir, trial_data_fname), 'rb') as f:
        trial_data = pkl.load(f)


    ### INITIALIZE NETWORK
    weight_list = weight_data['initial_final_weights']
    rnns = []

    for i, weights in enumerate(weight_list):
        wf = weights[1]

        rnn = rnnops.RNN(
            w_in=wf['W_in'].T,
            w_out=wf['W_out'].T,
            w_rec=wf['W_rec'].T,
            b=wf['b'],
            nonlinearity='relu',
            name='trained M-S network, l2 reg {}'.format(
                weight_data['l2_reg_scale'][i])
        )
        rnns.append(rnn)
        print(rnn)

    ### INITIALIZE TRIALS
    num_trials = len(trial_data)
    T = trial_data[0][0].shape[1]
    tt = np.arange(T, step=dt)

    def repeat_and_smooth(trial_data):
        return gaussian_filter1d(
            np.repeat(trial_data, int(1 / dt), axis=0), sigma=5, axis=0)

    def concat_trials(x):
        return np.moveaxis(np.array(x), 0, -1)

    inputs, targets = [], []
    for input_data, target_data, _, _ in trial_data:
        inp = repeat_and_smooth(np.moveaxis(input_data, 0, -1))
        tar = repeat_and_smooth(target_data.T)
        inputs.append(inp)
        targets.append(tar)
    inputs = concat_trials(inputs)
    targets = concat_trials(targets)


    def create_trials(batch_slice):
        trial = rnnops.trial.Trial(
            trial_len=T,
            dt=dt,
            name='Mante Sussillo all conditions',
            inputs=inputs[:, :, :, batch_slice],
            targets=targets[:, :, :, batch_slice]
        )
        return trial

    train_trials = create_trials(slice(0, num_train_trials))
    eval_trials = create_trials(
        slice(num_train_trials, num_train_trials + num_eval_trials))

    print(train_trials, train_trials.shape(1))
    print(eval_trials, eval_trials.shape(1))


    ### RUN BALANCING
    rnn_orig = rnns[-1]
    train_trials_orig = rnnops.trial.run_neural_dynamics(
        rnn_orig,
        train_trials,
        noise_std=0)
    cost_fn = robustness_cost_fn(
        train_trials_orig,
        nonlinearity='relu',
        weight_by_velocity=False,
    )
    rnn_balanced, opt_results = solve_balancing(
        rnn_orig,
        cost_fn=cost_fn,
        how='odeint',
        method='RK45',
        T_max=balancing_T_max,
    )
    print({k: opt_results[k] for k in ['c0', 'cf']})


    ### EVALUATE ORIGINAL AND BALANCED NETWORK WITH NOISY UNITS
    eval_trials_orig = [
        rnnops.trial.run_neural_dynamics(
            rnn_orig,
            eval_trials,
            noise_std=s
        ) for s in noise_levels]
    eval_trials_balanced = [
        rnnops.trial.run_neural_dynamics(
            rnn_balanced,
            eval_trials,
            noise_std=s
        ) for s in noise_levels]

    trials_orig_mean = [
        trial_apply(tr, np.mean) for tr in eval_trials_orig
    ]
    trials_balanced_mean = [
        trial_apply(tr, np.mean) for tr in eval_trials_balanced
    ]


    ### SAVE DATA FOR PLOTTING
    x_std_dev = np.std(eval_trials_orig[0].hiddens)
    trial_data = {
        'trials_orig_mean': trials_orig_mean,
        'trials_balanced_mean': trials_balanced_mean,
        'noise_levels': noise_levels,
        'balancing_results': opt_results,
        'x_std_dev': x_std_dev,
    }

    trial_data_path = join(data_dir, output_data_fname)
    with open(trial_data_path, 'wb') as f:
        pkl.dump(trial_data, f, -1)


if __name__ == '__main__':
    """
    When run from command line, find all runs without balancing and run 
    balancing on them.
    """

    project_dir = os.path.expanduser(
        '~/scratch/synaptic-balancing-PLOS/regularized_network_training'
    )
    run_prefix = 'N-256_context-noise-0.3_input-noise-0.3'
    dirs_for_processing = [
        f for f in os.listdir(project_dir)
        if (f.startswith(run_prefix)
        and not os.path.exists(
            os.path.join(project_dir, f, 'data', 'balanced_network.pkl')

        ))
    ]

    for run_dir in dirs_for_processing:
        run_balancing(
            os.path.join(project_dir, run_dir, 'data'),
            noise_levels=np.arange(0.3, step=0.03)
        )

