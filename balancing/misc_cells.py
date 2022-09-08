class TauCell(tf.contrib.rnn.RNNCell):
    """
    Cells with a variable time constant
    """
    def __init__(self, weights, n_steps=10, nonlinearity=tf.nn.tanh):
        """
            weights: dict containing weight matrices
            n_steps: int, 

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
        self.tau = tf.nn.sigmoid(tf.Variable(np.random.randn(self.num_neurons)*.1, name='tau'))

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
    
        # todo, something faster than for loop?
        for step in range(self.n_steps):
            x = x + (self.dt/self.tau)*(-x + tf.matmul(self.f(x), self.W_rec) + input_bias)

        return self.f(x), x

class CalciumCell(tf.contrib.rnn.RNNCell):
    def __init__(self, weights, n_steps=10, nonlinearity=tf.nn.tanh):
        """
            weights: dict containing weight matrices
            n_steps: int, 

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
    
        # todo, something faster than for loop?
        for step in range(self.n_steps):
            c_to_x = # fill in
            x_to_c = # fill in
            c = c + self.dt*(-c + tf.matmul(self.g(x), self.W_ca) + x_to_c)
            x = x + self.dt*(-x + tf.matmul(self.f(x), self.W_rec) + c_to_x + input_bias)

        return self.f(x), x

