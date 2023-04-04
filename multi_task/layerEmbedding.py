from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
import tensorflow as tf

class orderEmbedLayer(layers.Layer):
    def __init__(self, order_dict_size=1, embed_dim1=1, embed_reg=1, **kwargs):
        '''
        :param order_dict_size: number of items in shopping cart
        :param embed_dim1: output dimension of each item's embedding
        :param embed_reg: regularization number
        :param kwargs:
        :return: embedding vector for each item
        '''
        self.order_dict_size = order_dict_size
        self.embed_dim1 = embed_dim1
        self.embed_reg = embed_reg

        super(orderEmbedLayer, self).__init__(**kwargs)

        self.embed_order = layers.Embedding(input_dim = order_dict_size,
                                            output_dim = embed_dim1,
                                            embeddings_regularizer = regularizers.l2(embed_reg),
                                            embeddings_initializer = initializers.TruncatedNormal(stddev = 0.02))

    def call(self, order_feat, **kwargs):
        item_embedding = self.embed_order(order_feat[:, :-1])
        return item_embedding

    def get_config(self, ):
        config = {'order_dict_size': self.order_dict_size,
                  'embed_dim1': self.embed_dim1,
                  'embed_reg': self.embed_reg}
        base_config = super(orderEmbedLayer, self).get_config()

        return base_config.update(config)

class combAll(layers.Layer):
    def __init__(self, embed_dim1, **kwargs):
        '''
        :param embed_dim1: the size of each embedding
        :param kwargs:
        '''
        self.embed_dim1 = embed_dim1

        super(combAll, self).__init__(**kwargs)

    def call(self, x1, order_feat, **kwargs):
        real_len = order_feat[:, -1]
        _mask = tf.sequence_mask(lengths = real_len, maxlen = 7, dtype = tf.float32)
        _mask = tf.expand_dims(_mask, -1)
        mask = tf.tile(_mask, [1, 1, self.embed_dim])
        item_mask = x1 * mask
        item_pooling = tf.reduce_sum(item_mask, 1)
        item_count = tf.expand_dims(real_len, -1)
        item_avg_pooling = item_pooling / item_count

        return item_avg_pooling

    def get_config(self):
        config = {'embed_dim': self.embed_dim}
        base_config = super(combAll, self).get_config()

        return base_config.update(config)

class numEmbedLayer(layers.Layer):
    def __init__(self, onehot_dim, embed_dim2, embed_dim3, **kwargs):
        '''
        :param onehot_dim: one hot dimension
        :param embed_dim2: embedding dim of exposure items
        :param embed_dim3: embedding dim of numerical features
        :param kwargs:
        '''
        self.onehot_dim = onehot_dim
        self.embed_dim2 = embed_dim2
        self.embed_dim3 = embed_dim3

        super(numEmbedLayer, self).__init__(**kwargs)

        self.L11= layers.Dense(64)
        self.L12 = layers.Dense(embed_dim2)
        self.L21 = layers.Dense(64)
        self.L22 = layers.Dense(embed_dim3)

    def call(self, cate_feats, numerical_feats, **kwargs):
        exp_item = tf.cast(cate_feats[:, 0], tf.int32)
        exp_item_onehot = tf.one_hot(indices = exp_item - 1,
                                     depth = self.onehot_dim,
                                     axis = 1)
        item_embed = self.L11(exp_item_onehot)
        item_embed = tf.nn.relu(item_embed)
        item_embed = self.L12(item_embed)

        user_vector = layers.concatenate([cate_feats[:, 1:], numerical_feats], axis = 1)
        user_embed = self.L21(user_vector)
        user_embed = tf.nn.relu(user_embed)
        user_embed = self.L22(user_embed)

        cate_num_embed = layers.concatenate([item_embed, user_embed], axis = -1)

        return  cate_num_embed

    def get_config(self):
        config = {'onehot_dim': self.onehot_dim,
                  'embed_dim2': self.embed_dim2,
                  'embed_dim3': self.embed_dim3}
        base_config = super(numEmbedLayer, self).get_config()

        return base_config.update(config)

