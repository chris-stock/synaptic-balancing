# min_q_helper_2.py
# Helper functions for simulating a random line attractor and minimizing q 
from __future__ import division
import sys, datetime
from tqdm import trange, tqdm
import numpy as np
from numpy import random as npr
from scipy import optimize as opt
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD

# in the future, do this with tensorflow-  but for now it's fine

# Calcuate gradients, enforcing input dimensions
def f_explicit(x, u, rnn):
    # check inputs
    assert x.shape[0] == rnn['rec'].shape[1]
    assert x.shape[1] == 1

    # rnn dynamics
    r = rnn['network']['phi'](x)
    assert r.shape[1] == 1

    b = rnn['b'][:,None] if 'b' in rnn else 0

    f = np.array(-x + np.dot(rnn['rec'], r) + np.dot(rnn['in'], u)) + b 
    assert f.shape[1] == 1

    return f

def Df_explicit(x, u, rnn):
    # check inputs
    assert x.shape[0] == rnn['rec'].shape[1]
    assert x.shape[1] == 1

    # rnn dynamics
    f = f_explicit(x, u, rnn)
    assert f.shape[1] == 1

    Df_x = -np.eye(rnn['rec'].shape[0]) + rnn['rec'] * rnn['network']['phiprime'](x).T
 
    return Df_x

def q_explicit(x, u, rnn):
    # check inputs
    assert x.shape[0] == rnn['rec'].shape[1]
    assert x.shape[1] == 1

    f = f_explicit(x, u, rnn)
    assert f.shape[1] == 1

    q = np.sum(f ** 2) / 2
    assert len(q.shape) == 0

    return q

def Dq_explicit(x, u, rnn):
    # check inputs
    assert x.shape[0] == rnn['rec'].shape[0]
    assert x.shape[1] == 1

    # rnn dynamics
    f_x = f_explicit(x, u, rnn)
    assert f_x.shape[1] == 1

    Df_x = Df_explicit(x, u, rnn)
    assert Df_x.shape == rnn['rec'].shape

    # kinetic energy
    Dq_x = np.dot(Df_x.T, f_x)
    assert Dq_x.shape[1] is 1

    return Dq_x


def Hq_p_explicit(x, p, u, rnn):
    # check inputs
    assert x.shape[0] == rnn['rec'].shape[0]
    assert x.shape[1] is 1
    assert p.shape[0] == rnn['rec'].shape[0]
    assert p.shape[1] is 1

    # rnn dynamics    
    Df_x = Df_explicit(x, u, rnn)
    assert Df_x.shape == rnn['rec'].shape

    # compute Hessian applied to p
    Hq_p_x = np.dot(Df_x.T, np.dot(Df_x, p))
    assert Hq_p_x.shape[1] is 1

    return Hq_p_x

# Feed gradient into scipy opt
def find_one_ncg(x0, u, rnn, maxiter=256, tol=None, verbose=False):
    #import pdb; pdb.set_trace()
    u = u[:,None] if u.ndim==1 else u
    x0_mod = x0[:, 0] if x0.ndim>1 else x0
    q_mod = lambda x, u, rnn: q_explicit(x[:, None], u, rnn)
    Dq_mod = lambda x, u, rnn: Dq_explicit(x[:, None], u, rnn)[:, 0]
    Hq_p_mod = lambda x, p, u, rnn: Hq_p_explicit(x[:, None], p[:, None], u, rnn)[:, 0]

    if tol is None:
        tol = 1e-12

    x_min = opt.minimize(fun=q_mod, x0=x0_mod, args=(u, rnn),
                         jac=Dq_mod,
                         hessp=Hq_p_mod,
                         method='Newton-CG',
                         tol=tol,
                         options={'disp':verbose, 'maxiter': maxiter},
                         )

    return x_min #x_min['x'][:, None]


def find_many_randinit(num_samples, u, Wrec, Win, verbose=True):
    x0 = list()
    x_min = list()
    q_min = list()
    n = Wrec.shape[0]

    for i in range(num_samples):
        if verbose and i % 500 == 0:
            print("iter: ", str(i))

        # generate random initial condition
        x0_i = npr.rand() * npr.rand(n, 1)

        # minimize from initial condition
        x_min_i = find_one_ncg(x0_i, u, Wrec, Win)[:, 0]

        # save results of trial
        x0.append(x0_i[:,0])
        x_min.append(x_min_i)
        q_min.append(q_explicit(x_min_i[:, None], u, Wrec, Win))

    # cast as arrays
    x0 = np.array(x0).T
    x_min = np.array(x_min).T
    q_min = np.array(q_min)

    return x0, x_min, q_min


def find_many(samples, rnn, verbose=True, tol=None):
    # samples is a list of dicts with vector-valued keys 'r' and 'u'
    x_min = list()
    q_min = list()
    num_samples = len(samples)
    # print ""

    for i in trange(len(samples)):
        # if verbose and i % 10 == 0:
        #     print_over('iter: {:6d}/ {:6d}'.format(i, num_samples))

        # generate random initial condition
        x0_i = samples[i]['r'][:, None]
        u_i = samples[i]['u'][:, None]

        # minimize from initial condition
        x_min_i = find_one_ncg(x0_i, u_i, rnn, tol=tol)[:, 0]

        # save results of trial
        x_min.append(x_min_i)
        q_min.append(q_explicit(x_min_i[:, None], u_i, rnn))

    # cast as arrays
    x_min = np.array(x_min).T
    q_min = np.array(q_min)
    return x_min, q_min

def relu(x):
    return np.maximum(x, 0)

def corr(x1, x2):
    return np.corrcoef(x1.T, x2.T)[0,1]

# run dynamics 
def get_explored_states(num_trials, T, dt, randinit, u, Wrec, Win):
    n = Wrec.shape[0]
    
    # initalize dynamics across trials
    x_explored = np.zeros((n, T*dt, num_trials))
    if 'line' in randinit and randinit['line'] is not None:
        x_explored[:, 0, :] = randinit['line'][:, None] *  (randinit['mu'] + npr.randn(1, num_trials) * randinit['sigma'])
    else:
        x_explored[:, 0, :] = randinit['mu'] + randinit['sigma'] * npr.randn(n, num_trials)

    # run dynamics
    for i_tr in trange(num_trials):
        for t in range(1,T*dt):
            #import pdb; pdb.set_trace()
            x_prev = x_explored[:, t-1, i_tr]
            dx = Wrec.dot(relu(x_prev)) + (Win.dot(u))- x_prev
            assert dx.ndim is 1
            x_explored[:, t, i_tr] = x_prev + dx / float(dt)

    return x_explored

# i/o for writing over the same output line
def print_over(s):
    sys.stdout.write('\r')
    sys.stdout.flush()
    sys.stdout.write(s)
    sys.stdout.flush()
    pass

def timestamp():
    return '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now())


def plot_trial_all_networks(u_tr, x_tr_C, x_tr_T, z_tr_C, z_tr_T, s_tr, network):

    nfig = 6
    plt.figure(figsize=(12,18))
    nsteps = u_tr.shape[1]

    plt.subplot(nfig, 1, 1)
    plt.plot(u_tr.T)
    plt.ylim([-.2, 1.2])
    plt.xlim([0,nsteps])
    plt.title('FSM input')

    plt.subplot(nfig, 1, 2)
    plt.plot(x_tr_C.T)
    plt.xlim([0,nsteps])
    plt.title('Recurrent activity in constructed network (%d neurons)' % (network['nrec']))

    plt.subplot(nfig, 1, 3)
    plt.plot(x_tr_T.T)
    plt.xlim([0,nsteps])
    plt.title('Recurrent activity in trained network (%d neurons)' % (network['nrec']))

    plt.subplot(nfig, 1, 4)
    plt.imshow(z_tr_C, aspect='auto', interpolation='none', cmap='summer')
    plt.title('Output of constructed network')

    plt.subplot(nfig, 1, 5)
    plt.imshow(z_tr_T, aspect='auto', interpolation='none', cmap='summer')
    plt.title('Output of trained network')

    plt.subplot(nfig, 1, 6)
    plt.imshow(s_tr, aspect='auto', interpolation='none', cmap='summer')
    plt.title('Output of FSM (ground truth)')

    plt.xlabel('time steps')
    plt.show()

    pass

def plot_neural_trajectory(x_tr, u_tr, s_tr, x_fp=None):

    k_svd = 3
    svd = TruncatedSVD(k_svd)

    X = x_tr.T
    svd.fit(X - X.mean(0))
    axes = svd.components_
    tpcs = axes.dot(X.T)
    if x_fp is not None:
        fps = axes.dot(x_fp)

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')

    plt.show()

    for t in range(X.shape[0]):    

        plt.cla()

        plt.xlabel('PC1')
        plt.ylabel('PC2')

        plt.plot(tpcs[0,:], tpcs[1,:], tpcs[2,:], '.-', c='grey')
        if x_fp is not None:
            plt.plot(fps[0,:], fps[1,:], fps[2,:], 'o', c='salmon')

        if s_tr[0,t]==1:
            color = 'b'
        elif s_tr[1,t]==1:
            color = 'g'
        elif s_tr[2,t]==1:
            color = 'r'

        if u_tr[0,t]>0.5:
            size = 40
        else:
            size = 30

        plt.plot([tpcs[0,t]], [tpcs[1,t]], [tpcs[2,t]], '.', c=color, ms=size)
        plt.title('%d' % t)
        plt.draw()    
        plt.pause(0.02)
        
        pass