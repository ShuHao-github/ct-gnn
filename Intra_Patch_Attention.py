import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
# from tensorflow.keras.initializers import GlorotUniform, RandomUniform
import numpy as np

class GlorotUniform(tf.keras.initializers.Initializer):
    def __init__(self, seed=None):
        self.seed = seed

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = tf.keras.backend.floatx()

        fan_in, fan_out = self._compute_fans(shape)
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return tf.random.uniform(shape, -limit, limit, dtype=dtype, seed=self.seed)

    def get_config(self):
        return {'seed': self.seed}

    def _compute_fans(self, shape):
        if len(shape) == 2:  # Dense layer
            fan_in = shape[0]
            fan_out = shape[1]
        elif len(shape) == 4:  # Conv2D layer
            kernel_height, kernel_width, input_channels, output_channels = shape
            fan_in = kernel_height * kernel_width * input_channels
            fan_out = kernel_height * kernel_width * output_channels
        else:
            fan_in = np.prod(shape[:-1])
            fan_out = shape[-1]
        return fan_in, fan_out


class RandomUniform(tf.keras.initializers.Initializer):
    def __init__(self, minval=-0.05, maxval=0.05, seed=None):
        self.minval = minval
        self.maxval = maxval
        self.seed = seed

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = tf.keras.backend.floatx()
        return tf.random.uniform(shape, minval=self.minval, maxval=self.maxval, dtype=dtype, seed=self.seed)

    def get_config(self):
        return {'minval': self.minval, 'maxval': self.maxval, 'seed': self.seed}




class CustomLinear(Layer):
    def __init__(self, factorized, **kwargs):
        super(CustomLinear, self).__init__(**kwargs)
        self.factorized = factorized

    def build(self, inputs, weights, biases):
        # 确保 weights 和 biases 是具体的张量
        if isinstance(weights, (list, tuple)):
            weights = tf.convert_to_tensor(weights)
        if isinstance(biases, (list, tuple)):
            biases = tf.convert_to_tensor(biases)
        if self.factorized:
            # 如果是因子化的，需要在最后一个维度上增加一个维度，进行矩阵乘法后再压缩该维度
            inputs_expanded = tf.expand_dims(inputs, axis=-1)
            output = tf.matmul(inputs_expanded, weights)
            output = tf.squeeze(output, axis=-1) + biases
        else:
            # 如果不是因子化的，直接进行矩阵乘法
            output = tf.matmul(inputs, weights) + biases
        return output

class Intra_Patch_Attention(Layer):
    def __init__(self, d_model, factorized, **kwargs):
        super(Intra_Patch_Attention, self).__init__(**kwargs)
        self.head = 2

        if d_model % self.head != 0:
            raise Exception('Hidden size is not divisible by the number of attention heads')

        self.head_size = int(d_model // self.head)
        self.custom_linear = CustomLinear(factorized)

    def call(self, query, key, value, weights_distinct=None, biases_distinct=None, weights_shared=None, biases_shared=None):
        batch_size = query.shape[0]

        # key = self.custom_linear.build(key, weights_distinct[0], biases_distinct[0])
        # value = self.custom_linear.build(value, weights_distinct[1], biases_distinct[1])
        query = tf.concat(tf.split(query, self.head_size, axis=-1), axis=0)
        key = tf.concat(tf.split(key, self.head_size, axis=-1), axis=0)
        value = tf.concat(tf.split(value, self.head_size, axis=-1), axis=0)


        query = tf.transpose(query, perm=[1, 0, 2])
        key = tf.transpose(key, perm=[1, 2, 0])
        value = tf.transpose(value, perm=[1, 0, 2])


        attention = tf.matmul(query, key)
        attention /= tf.cast(tf.sqrt(tf.cast(self.head_size, tf.float32)), tf.float32)

        attention = tf.nn.softmax(attention, axis=-1)

        x = tf.matmul(attention, value)
        x = tf.transpose(x, perm=[1, 0, 2])
        x = tf.concat(tf.split(x, self.head_size, axis=0), axis=-1)

        # x = self.custom_linear.build(x, weights_shared[0], biases_shared[0])
        # x = tf.nn.relu(x)
        # x = self.custom_linear.build(x, weights_shared[1], biases_shared[1])
        return x, attention

class WeightGenerator(Layer):
    def __init__(self, in_dim, out_dim, mem_dim, num_nodes, factorized, number_of_weights=4, **kwargs):
        super(WeightGenerator, self).__init__(**kwargs)
        self.number_of_weights = number_of_weights
        self.mem_dim = mem_dim
        self.num_nodes = num_nodes
        self.factorized = factorized
        self.out_dim = out_dim

        if self.factorized:
            self.memory = self.add_weight(shape=(num_nodes, mem_dim), initializer='random_normal', trainable=True, name='memory')
            self.generator = tf.keras.Sequential([
                Dense(self.mem_dim, activation='tanh'),
                Dense(self.mem_dim, activation='tanh'),
                Dense(self.mem_dim ** 2)
            ])
            # self.mem_dim = 10
            self.P = [self.add_weight(shape=(in_dim, self.mem_dim), initializer=GlorotUniform(), trainable=True, name='P') for _ in range(number_of_weights)]
            self.Q = [self.add_weight(shape=(self.mem_dim, out_dim), initializer=GlorotUniform(), trainable=True, name='Q') for _ in range(number_of_weights)]
            self.B = [self.add_weight(shape=(self.mem_dim ** 2, out_dim), initializer=GlorotUniform(), trainable=True, name='B') for _ in range(number_of_weights)]
        else:
            self.P = [self.add_weight(shape=(in_dim, out_dim), initializer=GlorotUniform(), trainable=True, name='P') for _ in range(number_of_weights)]
            self.B = [self.add_weight(shape=(1, out_dim), initializer=RandomUniform(minval=-1.0, maxval=1.0), trainable=True, name='B') for _ in range(number_of_weights)]

    def call(self, x):
        if self.factorized:
            memory = self.generator(self.memory)
            bias = [tf.matmul(memory, self.B[i]) for i in range(self.number_of_weights)]
            print("memory shape:", memory.shape)
            print("bias shape:", bias)
            memory = tf.reshape(memory, (self.num_nodes, self.mem_dim, self.mem_dim))
            print('self.P',self.P)
            print("self.Q", self.Q)
            print("memory shape:", memory.shape)
            print([tf.matmul(self.P[i], memory) for i in range(self.number_of_weights)])
            weights = [tf.matmul(tf.matmul(self.P[i], memory), self.Q[i]) for i in range(self.number_of_weights)]
            return weights, bias
        else:
            return self.P, self.B


import tensorflow as tf
from tensorflow.keras import layers

class CrossAttention(layers.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dim = dim

        self.scale = qk_scale if qk_scale is not None else self.head_dim ** -0.5

        self.wq = layers.Dense(dim, use_bias=qkv_bias)
        self.wk = layers.Dense(dim, use_bias=qkv_bias)
        self.wv = layers.Dense(dim, use_bias=qkv_bias)
        self.attn_drop = layers.Dropout(attn_drop)
        self.proj = layers.Dense(dim)
        self.proj_drop = layers.Dropout(proj_drop)

    def call(self, x, x1):
        print('CROSS',self.dim,self.num_heads)
        B, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]

        # Query: only the first token
        q = self.wq(x[:, 0:1, :])  # B1C
        q = tf.reshape(q, [B, 1, self.num_heads, self.head_dim])  # B1C -> B1H(C/H)
        q = tf.transpose(q, [0, 2, 1, 3])  # B1H(C/H) -> BH1(C/H)

        # Key and Value: all tokens
        k = self.wk(x)  # BNC
        k = tf.reshape(k, [B, N, self.num_heads, self.head_dim])  # BNC -> BNH(C/H)
        k = tf.transpose(k, [0, 2, 1, 3])  # BNH(C/H) -> BHN(C/H)

        v = self.wv(x1)  # BNC
        v = tf.reshape(v, [B, N, self.num_heads, self.head_dim])  # BNC -> BNH(C/H)
        v = tf.transpose(v, [0, 2, 1, 3])  # BNH(C/H) -> BHN(C/H)

        # Attention calculation
        attn = tf.matmul(q, k, transpose_b=True) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        # Output calculation
        x = tf.matmul(attn, v)  # BH1N @ BHN(C/H) -> BH1(C/H)
        x = tf.transpose(x, [0, 2, 1, 3])  # BH1(C/H) -> B1H(C/H)
        x = tf.reshape(x, [B, 1, C])  # B1H(C/H) -> B1C

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


