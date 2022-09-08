"""
Recurrent Neural Net class
"""

import tensorflow as tf
from tqdm import trange
import numpy as np
from functools import partial

from .cells import ForwardEulerCell
from .components import randn_initializer, orth_initializer
from .utils import initialize_new_vars
from .optim import sgd_optimizer, balancing_optimizer

__all__ = ['BalancingRNN']

class BalancingRNN(object):

    def __init__(self,
                 task,
                 num_neurons,
                 Cell = ForwardEulerCell,
                 p = 2,
                 balancing_penalty_strength = 0.,
                 initializers = None,
                 eps_init = 1.,
                 sess = None,
                 num_timesteps = None,
                 **cell_kw
                 ):

        # store tensorflow session and task object
        self.sess = tf.Session() if sess is None else sess
        self.task = task
        self.num_neurons = num_neurons
        self.balancing_penalty_strength = balancing_penalty_strength

        # use task to initialize number of inputs and outputs
        self.num_inputs = task.num_inputs
        self.num_outputs = task.num_outputs        
        if num_timesteps is None:
            self.num_timesteps = task.num_timesteps
        else:
            self.num_timesteps = num_timesteps

        # defaults for weight matrix initialization
        initializers = {
            'W_rec': orth_initializer(eps_init),
            'W_in': randn_initializer(stddev=eps_init/np.sqrt(self.num_neurons)),
            'W_out': randn_initializer(stddev=eps_init/np.sqrt(self.num_neurons)),
            'b': lambda s, _: np.zeros(s)
        } if initializers is None else initializers

        # create weight matrices
        shapes = {
            'W_rec': (num_neurons, num_neurons),
            'W_in': (task.num_inputs, num_neurons),
            'W_out': (num_neurons, task.num_outputs),
            'b': (num_neurons,)
        }
        self.weights = {
            k: tf.Variable(initializers[k](shape, 0), name=k, dtype=tf.float32) for k, shape in shapes.items()
        }

        # define Balancing variables: cost matrix, cost function, and
        # imbalances. Here, a positive imbalance means outgoing costs are
        # larger than incoming costs
        self.balancing_p = p
        self.balancing_cost_mat = tf.pow(tf.abs(self.weights['W_rec']), self.balancing_p)
        self.balancing_cost = tf.reduce_sum(self.balancing_cost_mat)
        self.balancing_imbalances = tf.reduce_sum(self.balancing_cost_mat, axis=1, keepdims=True) \
                            - tf.transpose(tf.reduce_sum(self.balancing_cost_mat, axis=0, keepdims=True))
        self.balancing_dynamics = {
            'W_rec': tf.transpose(self.balancing_imbalances) * self.weights['W_rec'] \
                            - self.weights['W_rec'] * self.balancing_imbalances,
            'W_in': tf.transpose(self.balancing_imbalances) * self.weights['W_in'],
            'W_out': - self.weights['W_out'] * self.balancing_imbalances
        } 
        self.decay_dynamics = - self.balancing_p * tf.pow(tf.abs(self.weights['W_rec']), self.balancing_p - 1) \
                                           * tf.sign(self.weights['W_rec'])

        # create RNN cell
        self.cell = Cell(self.weights, **cell_kw)

        # create output weights
        self.W_out = self.weights['W_out']

        # placeholders for each batch
        self.target_placeholder = tf.placeholder(tf.float32, shape=([None]+task.target_dims))
        self.input_placeholder = tf.placeholder(tf.float32, shape=[None, self.num_timesteps, task.num_inputs])
        
        # rnn layer
        inputs = tf.unstack(self.input_placeholder, num=self.num_timesteps, axis=1)
        y, _ = tf.contrib.rnn.static_rnn(self.cell, inputs, dtype=tf.float32)
        self.emissions, self.states = zip(*y)
        self.outputs = tf.transpose(tf.stack([tf.matmul(e, self.W_out) for e in self.states]), (1, 2, 0))
        
        # loss
        self.loss = task.loss_function(self.outputs, self.target_placeholder)
        
        # regularization
        self.regularization = self.balancing_penalty_strength * self.balancing_cost
        
        # loss and objective function
        self.objective = tf.reduce_mean(self.loss) + self.regularization

        # create train op
        self.learning_rate = tf.placeholder(tf.float32, shape=[])

        initialize_new_vars(self.sess)

    @property
    def trainable_variables(self):
        return self.weights

    def assign_weights(self, param_dict):
        assign_ops = []
        for k in self.weights:
            if k in param_dict:
                assign_ops.append(tf.assign(self.weights[k], param_dict[k]))
        self.sess.run(assign_ops)

    def dump_weights(self):
        return self.sess.run(self.weights)

    def get_train_op(self, optimizer='balancing', **kwargs):

        clipping = {
            'method': tf.clip_by_global_norm,
            'args': (10.0,)
        }

        if optimizer=='sgd':
            train_op = sgd_optimizer(self, clipping=clipping, **kwargs)
        elif optimizer=='balancing':
            train_op = balancing_optimizer(self, clipping=clipping, **kwargs)

        return train_op


    def close_session(self):
        self.sess.close()

    def train(self, train_op, niter, learning_rate, probe_inputs=None, batch_size=1,
        trial_generator=None, append_to=None, save_weights=False, save_activity=False, save_balancing=True):
        
        if trial_generator is None:
            trial_generator = partial(self.task.generate_batch, batch_size=batch_size)
        
        # set train op
        self.train_op = train_op

        # things to fetch on each sess.run call
        fetches = {
            'train_op': self.train_op,
            'loss': self.loss,
            'regularization': self.regularization,
            'balancing_cost': self.balancing_cost,            
        }

        if save_activity:
            fetches.update({
                'outputs': self.outputs,
                'emissions': self.emissions,
                'states': self.states,
            })

        if save_balancing:
            fetches.update({
                'balancing_imbalances': self.balancing_imbalances,
                'balancing_dynamics_norm': tf.norm(self.balancing_dynamics['W_rec']),
                'decay_dynamics_norm': tf.norm(self.decay_dynamics),
            })

        # dictionary to compile training results
        if append_to is None:
            results = {k:[] for k in fetches}
            results.pop('train_op')
            results['targets'] = []
            results['weights'] = []
            results['probe_outputs'] = []
        else:
            results = append_to
            for k in 'loss', 'regularization', 'outputs', 'emissions', 'states':
                results[k] = list(results[k])

        # draw new training example
        probe_feed = {
            self.input_placeholder: probe_inputs,
        }
        probe_fetch = {
            'outputs': self.outputs
        }


        # main training loop
        pbar = trange(niter)
        for i in pbar:

            # draw new training example example
            inputs, target = trial_generator()
            feed_dict = {
                self.input_placeholder: inputs,
                self.target_placeholder: target,
                self.learning_rate: learning_rate
            }

            # run RNN and update parameters
            r = self.sess.run(fetches, feed_dict=feed_dict)
            for k in ['outputs', 'emissions', 'states', 'balancing_imbalances']:
                if k in r:
                    r[k] = np.squeeze(np.array(r[k]))
            r.pop('train_op')
            for k in r.keys():
                results[k].append(r[k])
            
            if save_weights:   
                results['targets'].append(target)
                results['weights'].append(self.dump_weights())                

            # update progressbar
            pbar.set_postfix(mean_loss=np.mean(r['loss']), balancing_cost=r['balancing_cost'])

            # run probe inputs
            if probe_inputs is not None:
                r = self.sess.run(probe_fetch, feed_dict=probe_feed)
                results['probe_outputs'].append(r['outputs'][-1])

        # transform the outputs (e.g. by softmax or sigmoids)
        if 'outputs' in results:
            results['outputs'] = self.task.transform(results['outputs'])
        
        # return everything as a numpy array
        for k in 'loss', 'balancing_cost', 'regularization', 'outputs', 'emissions', 'states':
            if k in results:
                results[k] = np.array(results[k])

        if probe_inputs is not None:
            results['probe_outputs'] = np.array(results['probe_outputs'])

        return results

    def simulate(self, batch_size=1, trial_generator=None, n_batch=1):
        """Simulate a batch from specified task

        Args
        ----
        task : Task object

        Returns
        -------
        states : numpy array, hidden states (timepoints x batch_size x neurons)
        outputs : numpy array, network output (timepoints x batch_size x outputs)
        loss: float, loss of the network on simulated trial
        """
        
        if trial_generator is None:
            trial_generator = partial(self.task.generate_batch, batch_size=batch_size)

        fetches = {
            'loss': self.loss,
            'outputs': self.outputs,
            'emissions': self.emissions,
            'states': self.states
        }

        
        if n_batch > 1:
            iterator = trange(n_batch)
        else:
            iterator = range(n_batch)
            
        result = {
            'loss': [],
            'outputs': [],
            'emissions': [],
            'inputs': [],
            'states': []
        }
        
        for b in iterator:
            inputs, target = trial_generator()
            feed_dict = {
                self.input_placeholder: inputs,
                self.target_placeholder: target
            }
            r = self.sess.run(fetches, feed_dict=feed_dict)
            result['inputs'].append(inputs.T)
            result['loss'].append(r['loss'])
            result['outputs'].append(np.squeeze(self.task.transform(np.array(r['outputs']))))
            result['emissions'].append(np.squeeze(np.array(r['emissions'])))
            result['states'].append(np.squeeze(np.array(r['states'])))
        result = {k: np.squeeze(v) for k, v in result.items()}
        return result, np.squeeze(target)
