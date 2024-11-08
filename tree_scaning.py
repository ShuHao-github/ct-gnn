import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Conv2D, Conv1D, ReLU, Softmax, Dropout, DepthwiseConv2D
import math
import numpy as np


def glorot(shape, scope='default', dtype=tf.float32):
    # Xavier Glorot & Yoshua Bengio (AISTATS 2010) initialization (Eqn 16)
    with tf.variable_scope(scope):
        init_range = np.sqrt(6.0 * INIT_SCALE / (shape[0] + shape[1]))
        init = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=dtype)
        return tf.Variable(init, dtype=dtype)


def zero_init(shape, scope='default', dtype=tf.float32):
    with tf.variable_scope(scope):
        init = np.zeros(shape)
        return tf.Variable(init, dtype=dtype)


class GraphAttentionLayer(object):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True, dtype=tf.float32):
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = glorot((self.in_features,self.out_features), dtype=dtype)
        self.a = glorot((2*self.out_features, 1), dtype=dtype)

    def build(self, h, adj):
        #print('GraphAttentionLayer')
        adj = tf.squeeze(tf.convert_to_tensor(tf.squeeze(adj, axis=0), dtype=tf.float32), axis=0)
        Wh = h @ self.W
        e = self._prepare_attentional_mechanism_input(Wh)

        # zero_vec = -9e15 * tf.ones_like(e)
        zero_vec = 0 * tf.ones_like(e)
        #print(adj, e, zero_vec)

        attention = tf.where(adj > 0, e, zero_vec)

        attention = tf.nn.softmax(attention, axis=1)
        attention = tf.nn.dropout(attention, self.dropout)

        #attention = attention * tf.cast(sph, dtype=attention.dtype)

        h_prime = tf.matmul(attention, Wh)

        if self.concat:
            return tf.nn.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = tf.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = tf.matmul(Wh, self.a[self.out_features:, :])
        e = tf.add(Wh1, tf.transpose(Wh2))
        return tf.nn.leaky_relu(e, self.alpha)


class GAT(object):
    def __init__(self,nheads, in_features, out_features, dropout, alpha, concat=True, dtype=tf.float32):
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(in_features, out_features, dropout, alpha, concat=True, dtype=tf.float32) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            setattr(self, f"attention_{i}", attention)

        self.out_att = GraphAttentionLayer(out_features * nheads, out_features, dropout, alpha, concat=False, dtype=tf.float32)

    def build(self, x, adj):
        x = tf.squeeze(tf.squeeze(x, axis=0), axis=0)
        x = tf.nn.dropout(x, keep_prob=self.dropout)
        x = tf.concat([att.build(x, adj) for att in self.attentions], axis=1)
        x = tf.nn.dropout(x, keep_prob=self.dropout)
        x = tf.nn.elu(self.out_att.build(x, adj))
        #x = tf.concat([att.build(x, adj) for att in self.attentions], axis=1)
        #x = tf.nn.dropout(x, keep_prob=self.dropout)
        #x = tf.nn.elu(self.out_att.build(x, adj))
        return tf.nn.log_softmax(x, axis=1)


class Softplus:
    def __init__(self):
        pass

    def forward(self, x):
        return tf.math.log(1 + tf.exp(x))


class LayerNormalization(Layer):
    def __init__(self,input_shape, axis=-1, epsilon=1e-6, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon
        self.gamma = self.add_weight(name='gamma', shape=input_shape, initializer='ones', trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape, initializer='zeros', trainable=True)

    def build(self, inputs):
        mean = tf.reduce_mean(inputs, axis=self.axis, keepdims=True)
        variance = tf.reduce_mean(tf.square(inputs - mean), axis=self.axis, keepdims=True)
        self.epsilon = tf.cast(self.epsilon, dtype=tf.float32)
        variance = tf.cast(variance, dtype=tf.float32)
        inputs = tf.cast(inputs, dtype=tf.float32)
        mean = tf.cast(mean, dtype=tf.float32)
        norm_inputs = (inputs - mean) * tf.squeeze(tf.math.rsqrt(variance + self.epsilon), axis=-1)
        return self.gamma * norm_inputs + self.beta

def batch_index_opr(data, index):
    channel = tf.shape(data)[1]
    index = tf.expand_dims(index, axis=1)
    index = tf.tile(index, [1, channel, 1])
    data = tf.gather(data, index, batch_dims=1)
    return data


def tree_scanning_core(xs, dts, As, Bs, Cs, Ds, delta_bias, origin_shape, h_norm):
    K = 1
    _, _, H, W = origin_shape
    B, D, L = xs.shape
    dts = tf.nn.softplus(dts + tf.expand_dims(tf.expand_dims(delta_bias, 0), -1))

    deltaA = tf.exp(dts * tf.expand_dims(As, 0))
    deltaB = tf.reshape(dts, [B, K, D // K, L]) * Bs
    BX = deltaB * tf.reshape(xs, [B, K, D // K, L])

    feat_in = tf.reshape(BX, [B, -1, L])
    edge_weight = deltaA

    def edge_transform(edge_weight, sorted_index, sorted_child):
        edge_weight = tf.gather(edge_weight, sorted_index, axis=1)
        return edge_weight,

    fea4tree_hw = tf.reshape(xs, [B, D, H, W])
    mst_layer = MinimumSpanningTree("Cosine", tf.exp)
    tree = mst_layer(fea4tree_hw)
    sorted_index, sorted_parent, sorted_child = bfs(tree, 4)
    edge_weight, = edge_transform(edge_weight, sorted_index, sorted_child)

    edge_weight_coef = tf.ones_like(sorted_index, dtype=edge_weight.dtype)
    feature_out = refine(feat_in, edge_weight, sorted_index, sorted_parent, sorted_child, edge_weight_coef)

    if h_norm is not None:
        out = h_norm(tf.transpose(feature_out, perm=[0, 2, 1]))

    y = tf.matmul(tf.reshape(out, [B, L, K, D]), tf.reshape(Cs, [B, L, K, -1]))
    y = tf.squeeze(y, axis=-1)
    y = tf.reshape(y, [B, -1, L])
    y = y + tf.reshape(Ds, [1, -1, 1]) * xs
    return y



INIT_SCALE = 3
DIFFERENT_H_STATES_RECURRENT_UPDATE_MECHANISM = True

class S6(object):
    def __init__(self, seq_len, d_model, state_size):
        self.seq_len = seq_len
        self.d_model = d_model
        self.state_size = state_size
        self.batch_size = 1

        self.fc1 = Dense(self.d_model)
        self.fc2 = Dense(self.state_size)
        self.fc3 = Dense(self.state_size)


        ones_tensor = tf.ones((self.d_model, self.state_size))
        normalized_tensor = tf.nn.l2_normalize(ones_tensor, axis=-1, epsilon=1e-12)
        self.A = tf.Variable(normalized_tensor)

        initializer = tf.glorot_uniform_initializer()
        self.A = tf.Variable(initializer(shape=(self.d_model, self.state_size)))


        self.B = tf.zeros((self.batch_size, self.seq_len, self.state_size))
        self.C = tf.zeros((self.batch_size, self.seq_len, self.state_size))

        self.delta = tf.zeros((self.batch_size, self.seq_len, self.d_model))
        self.dA = tf.zeros((self.batch_size, self.seq_len, self.d_model, self.state_size))
        self.dB = tf.zeros((self.batch_size, self.seq_len, self.d_model, self.state_size))

        # h [batch_size, seq_len, d_model, state_size]
        self.h = tf.zeros((self.batch_size, self.seq_len, self.d_model, self.state_size))
        self.y = tf.zeros((self.batch_size, self.seq_len, self.d_model))

    def discretization(self):
        # 假设self.delta的形状是(batch_size, length, depth)
        # 假设self.B的形状是(batch_size, length, num_features)
        # 假设self.A的形状是(num_features,)

        # 使用tf.einsum计算self.dB
        # "bld,bln->bldn" 表示第一个输入维度b与l对齐，第二个输入维度l与n对齐，d与第二个输入维度l对齐，并扩展一个新维度n
        self.dB = tf.einsum("bld,bln->bldn", self.delta, self.B)

        # 使用tf.einsum计算self.dA，然后应用exp函数
        # "bld,dn->bldn" 表示第一个输入维度b与l对齐，d与第二个输入维度d对齐，并扩展一个新维度n
        self.dA_unscaled = tf.einsum("bld,dn->bldn", self.delta, self.A)
        self.dA = tf.exp(self.dA_unscaled)

        # 返回计算得到的self.dA和self.dB
        return self.dA, self.dB


    def build(self, x, Adjacency_matrix):
        x = tf.squeeze(x, axis=0)
        self.B = self.fc2(x)
        self.C = self.fc3(x)
        self.delta = tf.nn.softplus(self.fc1(x))
        self.X = self.fc1(x)

        self.discretization()

        if DIFFERENT_H_STATES_RECURRENT_UPDATE_MECHANISM:
            # 获取当前批次大小
            current_batch_size = tf.shape(x)[0]

            # 检查当前批次大小是否与前一个时间步的隐藏状态大小不匹配
            if tf.shape(self.h)[0] != current_batch_size:
                different_batch_size = True

                h_truncated = self.h[:current_batch_size]

                h_new_unscaled = tf.einsum('bldn,blhf->bldh', self.dA, h_truncated)
                h_new_unscaled = h_truncated +self.dA


                x_expanded = tf.expand_dims(self.X, axis=-1)
                h_new = h_new_unscaled + x_expanded * self.dB
            else:
                different_batch_size = False
                h_truncated = self.h

                h_new_unscaled = tf.einsum('bldn,blhf->bldh', self.dA, h_truncated)
                h_new_unscaled = h_truncated + self.dA

                x_expanded = tf.expand_dims(self.X, axis=-1)

                h_new = h_new_unscaled + x_expanded * self.dB

            self.y = tf.einsum('bln,bldn->bld', self.C, h_new)

            global temp_buffer
            self.h.trainable = True
            temp_buffer = h_new if not self.h.trainable else tf.stop_gradient(h_new)

            return self.y

        else:
            # 初始化h和y为零张量，具有与x相同的设备位置
            h = tf.zeros((tf.shape(x)[0], self.seq_len, self.d_model, self.state_size), dtype=x.dtype)
            y = tf.zeros_like(x)

            h = tf.einsum('bldn,bldn->bldn', self.dA, h) + tf.expand_dims(x, axis=-1) * self.dB
            y = tf.einsum('bln,bldn->bld', self.C, h)

            return y


def tree_scanning(
        x=None,
        x_proj_weight=None,
        x_proj_bias=None,
        dt_projs_weight=None,
        dt_projs_bias=None,
        A_logs=None,
        Ds=None,
        out_norm=None,
        to_dtype=True,
        force_fp32=False,
        h_norm=None,
):
    B, D1, H, W = x.shape
    #print('B', B,'D', D1,'H', H,'W', W)
    origin_shape = x.shape
    #print('origin_shape', origin_shape)
    D2, N = A_logs.shape
    #print('D', D2, N)
    R = dt_projs_weight.shape[-1]  # 获取 R 的值
    #print('R', R)
    L = H * W
    #print('L',L)
    K =1


    xs = tf.reshape(x, [B, 1, D1, H * W])
    #print('xs shape', xs)
    #print('x_proj_weight shape', x_proj_weight)
    # if len(dt_projs_weight.shape) == 2:
    #     dt_projs_weight = tf.expand_dims(dt_projs_weight, axis=0)
    #
    # if len(x_proj_weight.shape) == 2:
    # x_proj_weight = tf.expand_dims(x_proj_weight, axis=-1)
    # x_proj_weight = tf.tile(x_proj_weight, [1, 1, 798])
    # x_proj_weight = tf.transpose(x_proj_weight, perm=[1, 0, 2])
    # print('x_proj_weight shape', x_proj_weight)
    #
    # x_dbl = tf.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    # if x_proj_bias is not None:
    #     x_dbl = x_dbl + tf.expand_dims(tf.expand_dims(x_proj_bias, 0), -1)
    x_dbl = xs
    R = tf.cast(R, tf.int32)
    N = tf.cast(N, tf.int32)
    dts, Bs, Cs = tf.split(x_dbl, [R, N, N], axis=2)
    #print('dts shape', dts, Bs, Cs)
    # dts = tf.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    xs = tf.reshape(xs, [B, -1, L])
    dts = tf.reshape(dts, [B, -1, L])
    As = -tf.exp(A_logs)
    Bs = tf.cast(Bs, tf.float32)
    Cs = tf.cast(Cs, tf.float32)
    Ds = tf.cast(Ds, tf.float32)
    delta_bias = tf.reshape(dt_projs_bias, [-1])

    if force_fp32:
        xs = tf.cast(xs, tf.float32)
        dts = tf.cast(dts, tf.float32)
        Bs = tf.cast(Bs, tf.float32)
        Cs = tf.cast(Cs, tf.float32)

    ys = tree_scanning_core(xs, dts, As, Bs, Cs, Ds, delta_bias, origin_shape, h_norm)
    ys = tf.reshape(ys, [B, K, -1, H, W])

    y = tf.reshape(ys, [B, K * -1, H * W])
    y = tf.transpose(y, perm=[0, 2, 1])
    y = out_norm(y)
    y = tf.reshape(y, [B, H, W, -1])

    return tf.cast(y, x.dtype) if to_dtype else y



class Conv2DWithGroups(Layer):
    def __init__(self, filters, kernel_size, groups=1, strides=(1, 1), padding='valid', use_bias=True, **kwargs):
        super(Conv2DWithGroups, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.groups = groups
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias

    def build(self, input_shape):
        self.depthwise_conv = DepthwiseConv2D(kernel_size=self.kernel_size,
                                              strides=self.strides,
                                              padding=self.padding,
                                              use_bias=self.use_bias)
        self.pointwise_conv = Conv2D(filters=self.filters,
                                     kernel_size=(1, 1),
                                     strides=(1, 1),
                                     padding='valid',
                                     use_bias=self.use_bias)

    def call(self, inputs):
        x = self.depthwise_conv(inputs)
        x = self.pointwise_conv(x)
        return x

    def get_config(self):
        config = super(Conv2DWithGroups, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'groups': self.groups,
            'strides': self.strides,
            'padding': self.padding,
            'use_bias': self.use_bias
        })
        return config

class RMSNorm(tf.keras.layers.Layer):
    def __init__(self, d_model: int, eps: float = 1e-5, **kwargs):
        super(RMSNorm, self).__init__(**kwargs)
        self.eps = eps
        self.weight = tf.Variable(initial_value=tf.ones(shape=(d_model,)), trainable=True)

    def call(self, inputs):
        rms = tf.math.sqrt(tf.reduce_mean(inputs ** 2, axis=-1, keepdims=True))
        output = inputs * tf.math.rsqrt(rms + self.eps) * self.weight

        return output


class Tree_SSM(Layer):
    def __init__(
        self,
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        ssm_rank_ratio=2.0,
        dt_rank="auto",
        act_layer=tf.keras.layers.ReLU,
        d_conv=3,
        conv_bias=True,
        dropout=0.0,
        bias=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        num_nodes=None,
        type=None,
        **kwargs,
    ):
        super(Tree_SSM, self).__init__(**kwargs)
        self.type = type
        # print("Tree_SSM init", d_model,d_state)
        d_expand = int(ssm_ratio * d_model)
        # print("Tree_SSM  d_expand:", d_expand)
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.dt_rank = np.ceil(d_model / 8) if dt_rank == "auto" else dt_rank
        self.d_state = np.ceil(d_model / 3) if d_state == "auto" else d_state

        self.d_conv = d_conv

        self.out_norm = LayerNormalization(d_model)
        self.h_norm = LayerNormalization(d_model)

        self.K = 1
        self.K2 = self.K

        #in_proj
        d_proj = d_expand * 2
        self.in_proj = Dense(d_proj, use_bias=bias)

        # conv
        if self.d_conv > 1:
            self.conv2d = Conv2DWithGroups(filters=d_expand, kernel_size=d_conv, groups=d_expand, padding='same',  use_bias=conv_bias)
        else:
            self.conv2d = Conv2DWithGroups(filters=d_expand, kernel_size=1, groups=1, padding='same',  use_bias=conv_bias)

        # x proj
        self.x_proj = Dense((self.dt_rank + self.d_state * 2), use_bias=False)
        self.x_proj.build((None, None, int(self.d_state)))
        self.x_proj_weight = self.x_proj.weights[0]

        # out proj
        self.out_proj = Dense(d_model, use_bias=bias)
        self.dropout = tf.keras.layers.Dropout(dropout) if dropout > 0.0 else tf.keras.layers.Lambda(lambda x: x)

        # dt proj
        self.dt_projs = Dense(self.dt_rank, use_bias=True)
        self.dt_projs.build((None, None,int(self.d_state)))
        self.dt_projs_weight = self.dt_projs.weights[0]
        self.dt_projs_bias = self.dt_projs.weights[1]

        # A, D
        self.A_logs = self.A_log_init(self.d_state, d_inner, copies=self.K2, merge=True)
        self.Ds = self.D_init(d_inner, copies=self.K2, merge=True)
        self.num_nodes = num_nodes

        self.linear = LinearAttention(dim=d_expand, num_heads=1, qkv_bias=True)
        self.gat = GAT(nheads=1, in_features=d_expand, out_features=d_expand,
                       dropout=dropout, alpha=0.2, concat=True, dtype=tf.float32)




    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = Dense(dt_rank, use_bias=True)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = tf.math.sqrt(1.0 / dt_rank) * dt_scale
        if dt_init == "constant":
            dt_proj.kernel.assign(tf.fill([d_inner, dt_rank], dt_init_std))
        elif dt_init == "random":
            dt_proj.kernel.assign(tf.random.uniform([d_inner, dt_rank], minval=-dt_init_std, maxval=dt_init_std))
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = tf.exp(tf.random.uniform([d_inner], minval=math.log(dt_min), maxval=math.log(dt_max))).clip_by_value(min=dt_init_floor)
        inv_dt = dt + tf.math.log(-tf.math.expm1(-dt))
        dt_proj.bias.assign(inv_dt)

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, merge=True):
        A = tf.range(1, d_state + 1, dtype=tf.float32)
        # print("A", A, d_inner,copies)
        A_log = tf.math.log(A)
        A_log = tf.tile(A_log,  multiples=[d_inner])
        # print("A_log", A_log)
        A_log = tf.reshape(A_log, [d_inner, d_state])
        if copies > 0:
            A_log = tf.expand_dims(A_log, axis=0)
            A_log = tf.tile(A_log,  multiples=[copies, A_log.shape[0], 1])
            if merge:
                A_log = tf.reshape(A_log, [-1, d_state])
        # print("A_log", A_log)
        return tf.Variable(A_log)

    @staticmethod
    def D_init(d_inner, copies=-1, merge=True):
        D = tf.ones(d_inner)
        if copies > 0:
            D = tf.tile(D,  multiples=[copies])
            if merge:
                D = tf.reshape(D, [-1])
        return tf.Variable(D)

    # def forward_core(self, x, channel_first=False, force_fp32=None):
    #     # if not channel_first:
    #     #     x = tf.transpose(x, perm=[0, 3, 1, 2])
    #     x = tree_scanning(
    #         x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
    #         self.A_logs, self.Ds,
    #         out_norm=self.out_norm,
    #         force_fp32=force_fp32,
    #         h_norm=self.h_norm
    #     )
    #
    #     return x

    def build(self, x, edge_index, edge_attr, **kwargs):
        x = self.in_proj(x)
        x, z = tf.split(x, num_or_size_splits=2, axis=-1)
        z = tf.keras.activations.relu(z)
        if self.d_conv > 1:
            x = tf.transpose(x, perm=[0, 3, 1, 2])
            x = self.conv2d(x)
        y = tf.keras.activations.relu(x)
        # print('y',y,z)
        ############################################### mamba
        if self.type == 'mamba':
            self.norm = RMSNorm(int(y.shape[3]))
            y = self.norm(y)
            self.ssm = S6(int(y.shape[2]), int(y.shape[3]), 7)  # int(4*y.shape[3])
            y = self.ssm.build(y, edge_attr)
            y = tf.expand_dims(y, axis=1)
            y = y * z
            out = self.dropout(self.out_proj(y))
        ################################################ LinearAttention
        elif self.type == 'linear':
            # y = x
            y = self.linear(y)
            y = y * z
            out = self.dropout(self.out_proj(y))
        ################################################ gat
        elif self.type == 'gat':
            # y = x
            y = self.gat.build(y, edge_attr)
            y = y * z
            out = self.dropout(self.out_proj(y))
        ############################################# ori
        else:
            y = y * z
            out = self.dropout(self.out_proj(y))
        return out

class Permute(Layer):
    def __init__(self, *args):
        super(Permute, self).__init__()
        self.args = args

    def call(self, x):
        return tf.transpose(x, perm=self.args)

class Mlp(Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=tf.keras.layers.ReLU, drop=0., channels_first=False):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        if channels_first:
            self.fc1 = Conv2D(filters=hidden_features, kernel_size=1, padding='valid')
            self.fc2 = Conv2D(filters=out_features, kernel_size=1, padding='valid')
        else:
            self.fc1 = Dense(hidden_features)
            self.fc2 = Dense(out_features)

        self.act = act_layer()
        self.drop = Dropout(drop)

    def call(self, x):
        if isinstance(x, tf.Tensor) and len(x.shape) == 4:  # Check if input is a 4D tensor
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
        else:
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
        x = self.drop(x)
        return x

class RoPE(tf.keras.layers.Layer):
    def __init__(self, shape, base=10000):
        super(RoPE, self).__init__()
        channel_dims, feature_dim = shape[:-1], shape[-1]
        k_max = feature_dim // (2 * len(channel_dims))
        print(feature_dim, k_max)
        assert feature_dim % k_max == 0

        theta_ks = 1 / (base ** (tf.range(k_max, dtype=tf.float32) / k_max))
        angles = tf.concat([tf.cast(t, tf.float32)[..., tf.newaxis] * theta_ks for t in
                            tf.meshgrid(*[tf.range(d) for d in channel_dims], indexing='ij')], axis=-1)

        rotations_re = tf.math.cos(angles)[..., tf.newaxis]
        rotations_im = tf.math.sin(angles)[..., tf.newaxis]
        rotations = tf.concat([rotations_re, rotations_im], axis=-1)
        self.rotations = tf.Variable(rotations, trainable=False)

    def call(self, x):
        if x.dtype != tf.float32:
            x = tf.cast(x, tf.float32)
        x = tf.complex(x[..., 0::2], x[..., 1::2])
        pe_x = tf.complex(self.rotations[..., 0], self.rotations[..., 1]) * x
        return tf.concat([tf.math.real(pe_x), tf.math.imag(pe_x)], axis=-1)

class LinearAttention(tf.keras.layers.Layer):
    def __init__(self, dim, input_resolution=None, num_heads=None, qkv_bias=True, **kwargs):
        super(LinearAttention, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.qk = Dense(dim * 2, use_bias=qkv_bias)
        self.elu = tf.keras.activations.elu
        self.lepe = Conv1D(dim, 3, padding='same')

    def call(self, x):
        print('SELF',self.dim, self.num_heads)
        self.rope = RoPE(shape=(tf.shape(x)[1], self.dim))

        # x = tf.squeeze(x, axis=0)
        b, n, c = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        num_heads = self.num_heads
        head_dim = c // num_heads

        qk = self.qk(x)
        q, k = tf.split(qk, 2, axis=-1)
        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0

        q_rope = self.rope(tf.reshape(q, [b, n, c]))
        k_rope = self.rope(tf.reshape(k, [b, n, c]))

        q = tf.reshape(q, [b, n, num_heads, head_dim])
        k = tf.reshape(k, [b, n, num_heads, head_dim])
        v = tf.reshape(x, [b, n, num_heads, head_dim])

        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        q_rope = tf.transpose(tf.reshape(q_rope, [b, n, num_heads, head_dim]), [0, 2, 1, 3])
        k_rope = tf.transpose(tf.reshape(k_rope, [b, n, num_heads, head_dim]), [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])

        z = 1 / (tf.matmul(q, tf.reduce_mean(k, axis=-2, keepdims=True), transpose_b=True) + 1e-6)
        kv = tf.matmul(k_rope, v, transpose_a=True) * (1.0 / tf.math.sqrt(tf.cast(n, tf.float32)))
        x = tf.matmul(q_rope, kv) * z

        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [b, n, c])

        v = tf.transpose(v, [0, 2, 1, 3])
        v = tf.reshape(v, [b, n, c])
        x = x + self.lepe(v)

        return x