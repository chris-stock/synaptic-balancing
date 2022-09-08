import tensorflow as tf
import numpy as np
import itertools
from ..utils import softmax, gram, matsqrt
from ..components import softmax_loss, rmse_loss

__all__ = ['SequenceClassificationTask']

_default_num_clusters = 8

_default_params = {
    'SNR': 20.,
    'tau': 1.,
    'ell': 3.,
    'num': 2,
    'seq_space': 2,
    'dt': 11,
    'seq_len': 19,
    'n_clusters': 6,
    'dtinv': 11,
    'kernel': 'matern1.5'
}

class SequenceClassificationTask(object):

    def __init__(self,
                 num_clusters = _default_num_clusters,
                 params = _default_params,
                 ):
        # params = _default_params.update(params)

        self.SNR = params['SNR']
        self.tau = params['tau']
        self.ell = params['ell']
        self.seq_space = params['seq_space']
        self.seq_len = params['seq_len']
        self.kernel = params['kernel']
        self.deltat = 1./params['dtinv']
        self.n_clusters = num_clusters

        clusters = [self._generate_cluster() for _ in range(self.n_clusters)]
        cluster_means, cluster_vars = zip(*clusters)
        self.clusters = {
            'mean': list(cluster_means),
            'sqrtvar': list(cluster_vars),
        }


    def generate_batch(self, batch_size=None, return_labels=False):
        inputs, targets, labels = [], [], []        
        if batch_size is None:
            nsamp = [1]*self.n_clusters
        else:
            nsamp = np.random.multinomial(batch_size, 
                [1./self.n_clusters]*self.n_clusters)
        for k in range(self.n_clusters):
            T = len(self.clusters['mean'][k])
            i = self.clusters['mean'][k][:,None] + \
                self.clusters['sqrtvar'][k].dot(
                    np.random.randn(T, nsamp[k]))
            t = [np.eye(self.n_clusters)[k]]*nsamp[k]            
            inputs.append(i.T)
            targets.extend(t)
            labels.extend([k]*nsamp[k])
        inputs = np.concatenate(inputs)[:,:,None] # batch_size x timepoints x 1
        targets = np.array(targets)  # batch_size x num_clusters
        labels = np.array(labels)
        if return_labels:
            return inputs, targets, labels
        else:
            return inputs, targets

    def loss_function(self, outputs, target):
        losses = tf.map_fn(lambda o: rmse_loss(o, target), tf.stack(outputs))
        return tf.reduce_mean(losses, axis=0)

    def generate_batch_constant_input(self, batch_size=1):
        inputs, targets, labels = [], [], []
        timepoints = len(self.clusters['mean'][0])
        inputs = np.random.rand(batch_size, 1)*np.ones((1, 10*timepoints, 1))
        inputs = 2.5*(2*inputs -1 )
        targets = np.zeros((batch_size, self.n_clusters))
        return inputs, targets    

    @property
    def num_inputs(self):
        return 1

    @property
    def num_outputs(self):
        return self.n_clusters

    @property
    def num_timesteps(self):
        return int(1+self.seq_space*(self.seq_len-1)/self.deltat)

    @property
    def target_dims(self):
        return [self.n_clusters]

    def transform(self, outputs):
        return outputs

    # kernel matrix function
    def _generate_cluster(self):
        # generate random discrete sequence for pinning gaussian processes
        t_obs = self.seq_space*np.arange(self.seq_len)
        tmin, tmax = t_obs[0], t_obs[-1]
        tt_seq = np.arange(tmin, tmax+self.deltat, step=self.deltat)
        tidx_obs = ((t_obs-tmin)/self.deltat).astype(int)
        y_raw = np.random.randn(1, self.seq_len-2)
        y_obs = np.concatenate([
            np.zeros((1,1)),
            y_raw,
            np.zeros((1, 1)),
            ], axis=1)[0]

        # calculate prior mean and covariance
        var_mu = gram(tt_seq, self.kernel, self.ell*self.seq_space)
        cov_y_mu = self.tau**2 * var_mu[tidx_obs, :]
        obs_noise = (y_obs != 0).astype(float)
        var_y = self.tau**2 * (cov_y_mu[:,tidx_obs] + np.diag(obs_noise)/self.SNR)
        proj = np.linalg.lstsq(var_y, cov_y_mu)[0] 
        post_E_mu = proj.T.dot(y_obs)
        post_var_mu = self.tau**2 * var_mu - proj.T.dot(cov_y_mu)        
        post_var_mu_sqrt = matsqrt(post_var_mu)

        return post_E_mu, post_var_mu_sqrt


