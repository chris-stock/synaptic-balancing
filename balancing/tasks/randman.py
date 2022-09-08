import numpy as np
from ..components import softmax_loss

__all__ = ['ManifoldClassification']


class ManifoldClassification(object):

    def __init__(self,
                 num_inputs, # number of inputs to the network, extrinsic dimensionality
                 intrinsic_dim, # intrinsic dimensionality of manifold
                 num_clusters, # number of clusters to separate
                 noise, # noise applied to each cluster
                 separation, # distance of each cluster from the origin
                 manifold_params = dict(), # params defining each manifold
                 cluster_ratios=None):

        self.num_inputs = num_inputs
        self.num_clusters = num_clusters
        self.num_outputs = num_clusters

        self.cluster_centers = np.random.randn(num_clusters, num_inputs) * separation

        manifold_params['dim'] = intrinsic_dim
        manifold_params['ext_dim'] = num_inputs

        self.manifolds = [SmoothManifold(**manifold_params) for _ in range(num_clusters)]

        if cluster_ratios is None:
            self.pvals = np.ones(num_clusters) / num_clusters
        else:
            assert all([r > 0 for r in cluster_ratios])
            self.pvals = np.array(cluster_ratios) / np.sum(cluster_ratios)

        if not np.iterable(noise):
            self.noise = np.full(num_clusters, noise)
        else:
            self.noise = noise

    def generate_batch(self, batch_size):
        # sample cluster ids
        N = np.random.multinomial(batch_size, pvals=self.pvals)

        X = np.empty((0, self.num_inputs))
        Y = np.empty((0, self.num_clusters))

        for i, n in enumerate(N):
            # sample n instances from cluster i
            x, _ = self.manifolds[i].sample(n)
            x += self.cluster_centers[i]
            x += self.noise[i] * np.random.randn(*x.shape)

            # generate labels
            y = np.zeros((n, self.num_clusters))
            y[:, i] = 1.0

            # concatenate to dataset
            X = np.row_stack((X, x))
            Y = np.row_stack((Y, y))

        return X, Y

    def loss_function(self, outputs, target):
        return softmax_loss(outputs, target)



class SmoothManifold(object):
    """Smooth manifold parameterized by a Fourier basis"""
    def __init__(self, dim=1, ext_dim=2, alpha=2, freq_cutoff=8):
        """
        Args:
            dim: intrinsic dimensionality
            ext_dim: extrinsic dimensionality
            alpha: exponent for power spectrum, 1/f**alpha
            freq_cutoff: frequency cutoff (XXX: what units?)
            phases: phase for each frequency component
            center: whether to center the manifold
        """
        self.dim = dim
        self.ext_dim = ext_dim
        self.alpha = alpha
        self.freq_cutoff = freq_cutoff
        self.phases = np.random.rand(ext_dim, dim, freq_cutoff)

        # Center manifold around the origin
        # Approximate the center of mass in extrinsic space
        self.mean = np.zeros((1, ext_dim))
        self.mean = self.sample(1000)[0].mean(0, keepdims=True)

    def get_params(self):
        """Get parameters of manifold"""
        return  {
            'dim': self.dim,
            'ext_dim': self.ext_dim,
            'alpha': self.alpha,
            'freq_cutoff': self.freq_cutoff,
            'phases': self.phases,
            'mean': self.mean,
        }
    
    def evaluate(self, y):
        x = np.ones((len(y), self.ext_dim))
        for e in range(self.ext_dim):
            for d in range(self.dim):
                xd = np.zeros(len(y))
                for f in range(self.freq_cutoff):
                    # Pink noise spectrum for amplitudes
                    S = 1.0 / (f + 1.0) ** self.alpha
                    xd  += S * np.sin(np.pi * f * y[:, d] + 2 * np.pi * self.phases[e, d, f])
                x[:, e] *= xd
        x -= self.mean
        return x

    def sample(self, n):
        y = np.random.rand(n, self.dim) * 2 * np.pi
        x = self.evaluate(y)
        return x, y
