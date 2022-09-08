"""
Helper functions for supreme adventuring
"""
import tensorflow as tf
import numpy as np

def compose(*funcs):
    """Composes a sequence of functions
    """
    def wrapper(x):
        for f in reversed(funcs):
            x = f(x)
        return x
    return wrapper

def softmax(x, axis=0):
    """Applies softmax function to numpy array
    """
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def gaussian(x, amplitude=1, stddev=1, mean=0):
    """Standard Gaussian function.
    Args:
        x: input vector
        amplitude: amplitude of the gaussian
        stddev: standard deviation of gaussian
        mean: center of gaussian
    Returns:
        y: applied gaussian function to each x
    """
    return amplitude * np.exp(-(x-mean)**2 / (2*stddev**2))

def multiplicative_gaussian_noise(x, stddev=1):
    """
    Args:
        x, ndarray: data without noise
    Returns:
        ndarray: data with added noise
    """ 
    return x + x*stddev*np.random.randn(*x.shape)

def no_noise(x):
    """Does not add noise to x
    """
    return x

def additive_gaussian_noise(x, stddev=1):
    """
    Args:
        x, ndarray: data without noise
    Returns:
        ndarray: data with added noise
    """
    return x + stddev*np.random.randn(*x.shape)

def poisson_noise(x):
    return np.random.poisson(x)

def gain_noise(x, *args):
    return x*np.random.lognormal(*args)

def get_uninitialized_vars(sess, var_list=None):
    """Gets a uninitialized variables in the current session
    Args:
        sess: a tensorflow session
        var_list (optional): list of variables to check. If None (default),
            then all global variables are checked
    Returns:
        uninitialized_vars: list of variables from var_list (or from the list
            of global variables if var_list was None) that have not yet
            been initialized
    """
    if var_list is None:
        var_list = tf.global_variables()
    is_init = sess.run(list(map(tf.is_variable_initialized, var_list)))
    return [v for v, init in zip(var_list, is_init) if not init]


def initialize_new_vars(sess, var_list=None):
    """Initializes any new, uninitialized variables in the session
    Args:
        sess: a tensorflow session
        var_list: list of variables to check (see `var_list` in
            `get_uninitialized_vars` for more information)
    """
    sess.run(tf.variables_initializer(get_uninitialized_vars(sess, var_list)))

def draw_wishart_cov(N, lam):
    """Draws a random NxN matrix L such that L.dot(L.T) is distributed as Wishart(lambda)
    Args:
        N: size of square matrix
        lam: Wishart parameter
    """ 
    P = round(N/lam)
    X = np.random.randn(N, P)
    X_U, X_s, _ = np.linalg.svd(X)
    if N > P:
        X_s = np.lib.pad(X_s,(0,N-P), mode='constant')
    L = X_U * X_s / np.sqrt(P)
    return L

def draw_powerlaw_cov(N, lam):
    """Draws a random NxN matrix L such that L.dot(L.T) is PSD with power law singular values
    satisfying s_i * lam = s_{i+1}
    Args:
        N: size of square matrix
        lam: base of power law
    """ 
    s = lam**np.arange(N)
    s = (N ** .5) * s/np.linalg.norm(s)
    U, _ = np.linalg.qr(np.random.randn(N, N))
    L = U * np.sqrt(s)

    return L

def draw_cov(N, cov_spec):
    if cov_spec[0]=='I' and len(cov_spec)>1:
        L = cov_spec[1]*np.eye(N)
    if cov_spec[0]=='Wish':
        L = draw_wishart_cov(N, lam=float(cov_spec[1]))
    elif cov_spec[0]=='powerlaw':
        L = draw_powerlaw_cov(N, lam=float(cov_spec[1]))
    else:
        L = np.eye(N)
    return L

def draw_Wrec(N, A_spec):
    # initialize recurrent network on edge of chaos
    return np.random.randn(N,N)/np.sqrt(N)


def gram(x, kernel='squared_exp', ell=1):            
    # given scalar input sequence x, generate gram matrix with specified kernel and length scale (ell)
    if kernel=='squared_exp':
        k = lambda d : np.exp(-d**2 / 2.)
    elif kernel=='matern1.5':
        k = lambda d : (1 + np.sqrt(3)*d) * np.exp(-np.sqrt(3)*d)
    elif kernel=='expo':
        k = lambda d : np.exp(-d)
    z = np.abs(x[:,None] - x[None,:])/ell
    C = k(z)/k(0)
    return C

def matsqrt(A):
    # snaps A to a symmetric PSD matrix and returns a matrix square root 
    A = (A + A.T)/2
    v, V = np.linalg.eigh(A)
    vpos = v*(v>0)
    return V * np.sqrt(vpos[None, :])