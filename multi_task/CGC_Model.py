import tensorflow as tf
from tensorflow.keras import layers, optimizers, regularizers

class CGClayer(layers.Layer):
    def __init__(self, expert_dim, n_expert, n_expert_share, n_task, dense_reg, **kwargs):
        '''
        :param expert_dim: the dim of expert layer
        :param n_expert: a list, each element in the list represents the number of experts required for each task
        :param n_expert_share: a list, each element in the list represent the number of the shared experts
        :param n_task: the number of tasks
        :param dense_reg: the regularization rule used in the model
        :param kwargs:
        '''
        self.expert_dim = expert_dim
        self.n_expert = n_expert
        self.n_expert_share = n_expert_share
        self.n_task = n_task
        self.dense_reg = dense_reg

        super(CGClayer, self).__init__(**kwargs)

        self.E_layer = []
        for i in range(n_task):
            sub_expert = [layers.Dense(expert_dim,
                                       activation = 'relu',
                                       kernel_regularizer = regularizers.l2(dense_reg))
                          for _ in range(n_expert[i])]
            self.E_layer.append(sub_expert)

        self.shared_layer = [layers.Dense(expert_dim,
                                          activation = 'relu',
                                          kernel_regularizer = regularizers.l2(dense_reg))
                             for _ in range(n_expert_share)]
        self.gate_layers = [layers.Dense(n_expert_share + n_expert[i], activation = 'softmax') for i in range(n_task)]

        def call(self, x, **kwargs):
            '''
            :param x: dense layer
            :return: a list, each element is a network for each task
            '''
            E_net = [[expert(x) for expert in sub_expert] for sub_expert in self.E_layer]
            share_net = [expert(x) for expert in self.shared_layer]

            towers = []
            for i in range(self.n_task):
                _gate = self.gate_layers[i](x)
                _gate = tf.expand_dims(_gate, axis = -1)
                _expert = share_net + E_net[i]
                _expert = layers.concatenate([expert[:, tf.newaxis, :] for expert in _expert], axis = 1) # tf.newaxis is used to add one dimension in experts
                _tower = tf.matmul(_expert, _gate, transpose_a = True)
                towers.append(layers.Flatten()(_tower))

            return towers

        def get_config(self,):
            config = {'expert_dim': self.expert_dim,
                      'n_expert': self.n_expert,
                      'n_expert_share': self.n_expert_share,
                      'n_task': self.n_task,
                      'dense_reg': self.dense_reg}
            base_config = super(CGClayer, self).get_config()

            config.update(base_config)

            return config