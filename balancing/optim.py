"""
Convienence functions for optimizers
"""
import tensorflow as tf
from .utils import initialize_new_vars

__all__ = ['sgd_optimizer', 'balancing_optimizer']

def clip_gradients(opt, network, clipping, var_list):
    # by default, train all variables
    if var_list is None:
        var_list = list(network.trainable_variables.values())

    # if desired clip gradients
    if clipping is None:
        train_op = opt.minimize(network.objective, var_list=var_list)
    else:
        assert type(clipping) == dict
        gradients, variables = zip(*opt.compute_gradients(network.objective, var_list=var_list))
        gradients, _ = clipping['method'](gradients, *clipping['args'])
        train_op = opt.apply_gradients(zip(gradients, variables))

    # initialize anything new created by the optimizer
    initialize_new_vars(network.sess)
    return train_op


def sgd_optimizer(network, clipping=None, var_list=None, **kwargs):
    """Returns a standard Gradient Descent operator with gradient clipping.
    """
    opt = tf.train.GradientDescentOptimizer(network.learning_rate, **kwargs)
    return clip_gradients(opt, network, clipping, var_list)

def balancing_optimizer(network, grad_rate=1., balancing_rate=0., decay_rate=0.,
                  clipping=None, var_list=None, **kwargs):
    """Returns a Synaptic Balancing optimizer for a recurrent network.
    """
    
    sgd = tf.train.GradientDescentOptimizer(network.learning_rate, **kwargs)
    
    # by default, train all variables
    if var_list is None:
        var_list = list(network.trainable_variables.values())

    # get gradients
    gradients, variables = zip(*sgd.compute_gradients(network.objective, var_list=var_list))

    # if desired, clip gradients
    if clipping is not None:
        assert type(clipping) == dict
        gradients, _ = clipping['method'](gradients, *clipping['args'])

    # calculate balancing update
    grad_rate = tf.convert_to_tensor(grad_rate, dtype=tf.float32)
    balancing_rate = tf.convert_to_tensor(balancing_rate, dtype=tf.float32)
    updates = []
    for gradient, variable in zip(gradients, variables):
        for vname in ['W_rec', 'W_in', 'W_out']:
            if variable.name.startswith(vname):
                update = grad_rate*gradient - balancing_rate*network.balancing_dynamics[vname]
                updates.append((update, variable))
    train_op = sgd.apply_gradients(updates)

    # initialize anything new created by the optimizer
    initialize_new_vars(network.sess)

    return train_op