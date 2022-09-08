import tensorflow as tf


class ForwardEulerCell(tf.contrib.rnn.RNNCell):
    def __init__(
            self,
            weights,
            n_steps=10,
            report_every=None,
            nonlinearity=tf.nn.tanh
    ):
        """
            weights: dict, containing weight matrices
            n_steps: int, number of integration time steps
            nonlinearity: function, firing rate nonlinearity (e.g. tanh or relu)
        """
        self.W_rec = weights['W_rec']
        self.W_in = weights['W_in']
        self.b = weights['b']
        self.f = nonlinearity
        self.num_neurons = self.W_rec.get_shape().as_list()[0]
        if type(n_steps) is not int:
            raise ValueError('n_steps must be specified as int.')
        self.n_steps = n_steps
        self.dt = 1.0 / n_steps
        if report_every is None:
            self.report_every = self.n_steps
        else:
            self.report_every = report_every

    @property
    def state_size(self):
        return self.num_neurons

    @property
    def output_size(self):
        return self.num_neurons

    def __call__(self, u, x, scope=None):
        """
        Args:
            u : inputs (batch_size x num_inputs)
            x : last state (batch_size x state_size)
        """
        
        # input to network and bias term
        input_bias = tf.matmul(u, self.W_in) + self.b

        def fn(x_, t):
            # one a forward Euler integration step of the neural dynamics
            return x_ + self.dt*(
                    -x_ + tf.matmul(self.f(x_), self.W_rec) + input_bias
            )

        # roll up multiple forward Euler steps into a single op
        out = tf.scan(fn, tf.range(self.report_every), initializer=x)

        return (self.f(out[-1]), out[-1]), out[-1]