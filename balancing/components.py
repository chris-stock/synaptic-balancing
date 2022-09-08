"""
Nonlinearities, Loss functions, Regularizers, Initializers (many are aliases)
"""
import tensorflow as tf
import numpy as np


# Parameter Initialization
##########################

def randn_initializer(mean=0.0, stddev=1.0):
    """Generates a function for random Gaussian initialization
    """
    def _init(shape, i):
        return np.random.randn(*shape).astype(np.float32)*stddev + mean
    return _init


def orth_initializer(scale=1.):
    """Generates a function for random orthogonal matrix initialization
    """
    def _init(shape, i):
        q, r = np.linalg.qr(np.random.randn(*shape))
        return (scale*q).astype(np.float32)
    return _init


def identity_initializer(scale=1., *args, **kwargs):
    """Generates a function for identity matrix initialization
    """ 
    return lambda shape, i: scale*np.eye(*shape).astype(np.float32)


def custom_initializer(W_init, *args, **kwargs):
    """Generates a function for custom initialization of each weight matrix
    """
    def _init(shape, i):
        try:
            W_i = W_init[i]
        except IndexError:
            raise IndexError('W_init must have as many elements as there are'
                             'weight matrices')
        assert W_i.shape == shape, 'W_init[{}] has shape {}, should have' \
                                   'shape {}'.format(i, W_init.shape, shape)
        return W_i.astype(np.float32)
    return _init


# Regularizers
##############
l1_regularizer = tf.contrib.layers.l1_regularizer
l2_regularizer = tf.contrib.layers.l2_regularizer


def rms_regularizer(s):
    return lambda x: s * tf.sqrt(tf.reduce_mean(x**2))

def frob_regularizer(s):
    return lambda x: s * tf.reduce_mean(x**2)

def no_regularizer(x):
    return tf.constant(0, dtype=tf.float32)


# Nonlinearities
################
identity = tf.identity
relu = tf.nn.relu
tanh = tf.nn.tanh
square = tf.square


# Loss Functions
################
def softmax_loss(outputs, target):
    return tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=target)


def rmse_loss(outputs, target):
    return tf.sqrt(tf.reduce_mean((outputs-target)**2))


# Metrics
################
def accuracy(outputs, target):
    predictions = tf.argmax(outputs, axis=1)
    labels = tf.argmax(target, axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))
