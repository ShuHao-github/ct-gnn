import datetime
import random

import networkx as nx
import numpy as np
import tensorflow as tf
import traceback
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import FastICA
from sklearn.decomposition import NMF
from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA
import math
from tensorflow.keras import layers
from collections import defaultdict
from itertools import chain
from transformer import TransformerEncoderLayer
from st_gnn import ST_GNN_Model

from COGNN import CoGNN
from CAGNN import CAGNN
from TreeScan import TreeScan
from tree_scaning import LinearAttention
from Intra_Patch_Attention import Intra_Patch_Attention, WeightGenerator, CrossAttention


from mamba import RMSNorm, MambaBlock
from dgnn import DGNN


#from model.transform import data

#position == 1:"PGNN"
#position == 2:"PADEL"
#position == 3:"PADEL_ALL"

# nn_primitives_ak123_padel+pe_pca

INIT_SCALE = 1


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


class SingleLayerFNN(object):
    def __init__(self, inp_size, inp_shape, name):
        self.w = glorot(inp_shape, name)
        self.b = zero_init((1, inp_size), name)

    def build(self, input_tensor):
        out = input_tensor
        out = tf.matmul(out, self.w) + self.b
        out = tf.nn.relu(out)
        return out


class FNN(object):
    # hidden_layer_sizes: list of hidden layer sizes
    # out_size: Size of the last softmax layer
    def __init__(self, inp_size, hidden_layer_sizes, out_size, name, dtype=tf.float32):

        layers = []
        sizes = [inp_size] + hidden_layer_sizes + [out_size]
        for i in range(len(sizes) - 1):
            w = glorot((sizes[i], sizes[i + 1]), name, dtype=dtype)
            b = zero_init((1, sizes[i + 1]), name, dtype=dtype)
            layers.append([w, b])

        self.layers = layers

    # *Don't* add softmax or relu at the end
    def build(self, inp_tensor):
        out = inp_tensor
        for idx, [w, b] in enumerate(self.layers):
            out = tf.matmul(out, w) + b
            if idx != len(self.layers) - 1:
                out = tf.nn.relu(out)

        return out

class GraphConvolution(object):
    def __init__(self, inp_features, out_features, bias_select=False, dtype=tf.float32):
        self.inp_features = inp_features
        self.out_features = out_features
        self.bias_select = bias_select
        self.weight = glorot(self.out_features, dtype=dtype)
        self.bias = zero_init((1, self.out_features[1]), dtype=dtype)

    def build(self, input, adj):
        support = tf.matmul(input, self.weight)
        output = tf.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GCN(object):
    def __init__(self, inp, hid, out, dropout):
        self.gc1 = GraphConvolution(inp, hid)
        #self.gc2 = GraphConvolution(hid, out)
        self.gc2 = GraphConvolution(inp, out)
        self.dropout = dropout

    def build(self, x, adj):
        x = tf.nn.relu(self.gc1.build(x, adj))
        x = tf.nn.dropout(x, keep_prob=self.dropout)
        #x =self.gc2.build(x, adj)
        x = tf.nn.log_softmax(x)
        #x = tf.nn.dropout(x, keep_prob=self.dropout)
        return x



class GraphAttentionLayer(object):
    def __init__(self, in_features, out_features, dropout, alpha, type=None, concat=True, dtype=tf.float32):
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = glorot((self.in_features,self.out_features), dtype=dtype)
        self.a = glorot((2*self.out_features, 1), dtype=dtype)
        self.type = type

    def build(self, h, adj, sph):
        Wh = h @ self.W
        e = self._prepare_attentional_mechanism_input(Wh)

        # zero_vec = -9e15 * tf.ones_like(e)
        zero_vec = 0 * tf.ones_like(e)

        attention = tf.where(adj > 0, e, zero_vec)

        attention = tf.nn.softmax(attention, axis=1)
        attention = tf.nn.dropout(attention, self.dropout)

        if sph is not None:
            if self.type == "mul":
                attention = attention * tf.cast(sph, dtype=attention.dtype)
            elif self.type == "add":
                attention = attention + tf.cast(sph, dtype=attention.dtype)

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
    def __init__(self,nheads, in_features, out_features, dropout, alpha, type=None, concat=True, dtype=tf.float32):
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(in_features, out_features, dropout, alpha, type=type, concat=True, dtype=tf.float32) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            setattr(self, f"attention_{i}", attention)

        self.out_att = GraphAttentionLayer(out_features * nheads, out_features, dropout, alpha, type=type, concat=False, dtype=tf.float32)

    def build(self, x, adj, sph=None):
        x = tf.nn.dropout(x, keep_prob=self.dropout)
        x = tf.concat([att.build(x, adj, sph) for att in self.attentions], axis=1)
        x = tf.nn.dropout(x, keep_prob=self.dropout)
        x = tf.nn.elu(self.out_att.build(x, adj, sph))
        #x = tf.concat([att.build(x, adj) for att in self.attentions], axis=1)
        #x = tf.nn.dropout(x, keep_prob=self.dropout)
        #x = tf.nn.elu(self.out_att.build(x, adj))
        return tf.nn.log_softmax(x, axis=1)


def swish(x):
    return x * tf.keras.backend.sigmoid(x)

def get_activation(act):
    if act == "silu":
        return tf.keras.layers.Lambda(swish) # TensorFlow中使用'swish'代替'silu'
    elif act == "leaky_relu":
        return tf.keras.layers.LeakyReLU(alpha=0.01)  # 可以根据需要调整alpha值
    else:
        raise ValueError(f"Unsupported activation: {act}")


class BaseConv(tf.keras.layers.Layer):
    """A Conv2D -> BatchNorm -> silu/leaky relu block"""

    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super(BaseConv, self).__init__()
        self.conv = tf.keras.layers.Conv1D(
            filters=out_channels,
            kernel_size=ksize,
            strides=stride,
            padding='same',  # TensorFlow中'same'实现与PyTorch中相同填充效果
            use_bias=bias
        )
        self.bn = tf.keras.layers.BatchNormalization()
        self.act = get_activation(act)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        return self.act(x)

    def fuse_forward(self, inputs):
        x = self.conv(inputs)
        return self.act(x)

class Focus(object):
    def __init__(self, size, out_feature=7,in_channels=1, out_channels=1, ksize=1, stride=1, act="silu", dtype=tf.float32):
        self.size = size
        self.x_col_size = self.size[1]
        self.x_row_size = self.size[0]
        if self.x_col_size % 2 == 0:
            self.x_size = self.x_col_size
        else:
            self.x_size = self.x_col_size + 1
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

        #self.W = glorot((int(self.x_size/2), self.out_features), dtype=dtype)

    def build(self, x):
        x = tf.keras.layers.Dense(self.x_size)(x)

        if self.x_row_size % 2 != 0:
            zeros_tensor = tf.zeros([1, self.x_size])
            x = tf.concat([x, zeros_tensor], axis=0)

        x = tf.expand_dims(x, axis=-1)
        patch_top_left = x[::2, ::2, ...]
        patch_top_right = x[::2, 1::2, ...]
        patch_bot_left = x[1::2, ::2, ...]
        patch_bot_right = x[1::2, 1::2, ...]
        patch_x = tf.concat(
            [
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ],
            axis=-1
        )
        patch_x = self.conv(patch_x)
        patch_x = tf.squeeze(patch_x, axis=-1)

        return patch_x

class I2GCN(object):
    def __init__(self,in_features, embedding_size_deg, num_layers=5, subgraph_pooling='mean', subgraph2_pooling='mean',
                 use_pooling_nn=False,use_rd=False,double_pooling=False, gate=False):
        assert (subgraph_pooling=='mean' or subgraph_pooling=='add' or subgraph_pooling=='mean-context') \
               and (subgraph2_pooling=='mean' or subgraph2_pooling=='add' or subgraph2_pooling=='center'
                    or subgraph2_pooling=='mean-center' or subgraph2_pooling=='mean-center-side')
        self.in_features = in_features
        self.num_layers = num_layers
        self.subgraph_pooling = subgraph_pooling
        self.subgraph2_pooling =subgraph2_pooling
        s2_dim = 2 if subgraph2_pooling=='mean-center' else 3 if subgraph2_pooling=='mean-center-side' else 1
        s2_dim = s2_dim + 1 if subgraph_pooling == 'mean-context' else s2_dim
        self.use_rd = use_rd
        if self.use_rd:
            self.rd_projection = tf.keras.layers.Dense(8)

        self.node_type_embedding = SingleLayerFNN(inp_size=8, inp_shape=(100,8), name='positional_awareness')
        self.double_pooling = double_pooling
        self.res = True
        self.gate = gate
        if self.gate:
            self.subgraph_gate = []
        self.new_x = []
        self.embedding_size_deg = embedding_size_deg

        # node label feature
        self.convs = []
        self.norms = []
        self.z_embedding_list =  []
        if self.use_rd:
            self.rd_projection_list = []

        if self.double_pooling:
            self.double_nns = []
        M_in, M_out = self.embedding_size_deg+2, 64
        self.GINConv = SingleLayerFNN(inp_size=M_out, inp_shape=(2*M_in,M_out), name='positional_awareness')
        self.fnn_rew_x_2 = SingleLayerFNN(inp_size=2, inp_shape=(self.embedding_size_deg, 2), name='positional_awareness')


        #first layer  157
        #self.convs.append(SingleLayerFNN(inp_size=M_out, inp_shape=(2*M_in,M_out), name='positional_awareness'))
        self.convs.append(tf.keras.layers.Dense(M_out))
        self.z_embedding_list.append(tf.keras.layers.Dense(M_in))
        if self.use_rd:
            self.rd_projection_list.append( tf.keras.layers.Dense(M_in))
        if self.gate:
            self.subgraph_gate.append(tf.keras.layers.Dense(M_out))   #activation='sigmiod'
            self.subgraph_gate.append(tf.keras.layers.Activation('sigmoid'))
        self.norms.append(tf.keras.layers.BatchNormalization())
        if self.double_pooling:
            self.double_nns.append(tf.keras.layers.Dense(128))   #activation='relu'
            self.double_nns.append(tf.keras.layers.Activation('relu'))
            self.double_nns.append(tf.keras.layers.Dense(M_out))

        for i in range(num_layers-1):
            M_in, M_out = M_out, 64
            self.z_embedding_list.append(tf.keras.layers.Dense(M_in))
            if self.use_rd:
                self.rd_projection_list.append(tf.keras.layers.Dense(M_in))
            if self.gate:
                self.subgraph_gate.append(tf.keras.layers.Dense(M_out)) #activation='sigmiod'
                self.subgraph_gate.append(tf.keras.layers.Activation('sigmoid'))

            self.convs.append(tf.keras.layers.Dense(M_out))
            self.norms.append(tf.keras.layers.BatchNormalization())

            if self.double_pooling:
                self.double_nns.append(tf.keras.layers.Dense(128))  #activation='relu'
                self.double_nns.append(tf.keras.layers.Activation('relu'))
                self.double_nns.append(tf.keras.layers.Dense(M_out))

        # MLPs for hierarchical pooling
        self.egde_pooling_nn = []
        self.node_pooling_nn = []
        if use_pooling_nn:
            self.egde_pooling_nn.append(
                tf.keras.layers.Dense(s2_dim * M_out))  #activation='relu'
            self.egde_pooling_nn.append(
                tf.keras.layers.Activation('relu'))  # activation='relu'
            self.egde_pooling_nn.append(
                tf.keras.layers.Dense(s2_dim * M_out))

            self.node_pooling_nn.append(
                tf.keras.layers.Dense(s2_dim * M_out))  # activation='relu'
            self.node_pooling_nn.append(
                tf.keras.layers.Activation('relu'))  # activation='relu'
            self.node_pooling_nn.append(
                tf.keras.layers.Dense(s2_dim * M_out))
        self.use_pooling_nn = use_pooling_nn

        #final graph pooling
        if self.use_rd:
            self.rd_projection_list.append(tf.keras.layers.Dense(M_out))
        if self.gate:
            self.subgraph_gate.append(tf.keras.layers.Dense(M_out))#activation='sigmiod'
            self.subgraph_gate.append(tf.keras.layers.Activation('sigmoid'))

        self.fc1 = tf.keras.layers.Dense(32)
        self.fc2 = tf.keras.layers.Dense(16)
        self.fc3 = tf.keras.layers.Dense(self.embedding_size_deg)

        self.gat = GAT(nheads=2, in_features=2*M_in, out_features=M_out,
                       dropout=0.7, alpha=0.2, concat=True, dtype=tf.float32)

        self.gat0 = GAT(nheads=1, in_features=9, out_features=64, dropout=0.7, alpha=0.7, concat=True, dtype=tf.float32)

        self.gat1 = GAT(nheads=1, in_features=64, out_features=64, dropout=0.7, alpha=0.7, concat=True, dtype=tf.float32)


    def center_pool(self, x, indices, num_segments=None):
        global_feat = tf.reduce_mean(x, axis=0)
        center_feat = x - global_feat
        subgraph_feat = center_feat * tf.one_hot(64, 64)
        subgraph_feat = tf.math.unsorted_segment_sum(subgraph_feat, indices, num_segments)

        return subgraph_feat

    def graph_pooling(self, x, z, layer=None, aggr='mean', node_emb_only=False, node_to_original_node=None, node_to_subgraph2=None, num=None, center_idx=None):

        #print("graph_pooling")
        # functions = [func for func in dir(tf.keras.layers) if callable(getattr(tf.keras.layers, func))]
        # print(functions)
        if self.subgraph_pooling == 'mean-context':
            #print('mean-context')
            x_node = tf.math.unsorted_segment_mean(x, node_to_original_node, num_segments=num)

        if self.subgraph2_pooling == 'mean':
            #print("mean")
            if self.gate:
                z = self.subgraph_gate[2 * layer](z)
                x = self.subgraph_gate[2 * layer +1](z) * x
            x = tf.math.unsorted_segment_mean(x, node_to_subgraph2, num_segments=num)
        elif self.subgraph2_pooling == 'add':
            #print('add')
            x = tf.math.unsorted_segment_sum(x, node_to_subgraph2, num_segments=num)
        elif self.subgraph2_pooling == 'center':
            #print('center')
            x = self.center_pool(x, node_to_subgraph2, num_segments=num)
        elif self.subgraph2_pooling == 'mean-center':
            #print('mean-center')
            x = tf.concat((tf.math.unsorted_segment_mean(x, node_to_subgraph2, num_segments=num),
                           self.center_pool(x, node_to_subgraph2, num_segments=num)), axis=-1)
        elif self.subgraph2_pooling == 'mean-center-side':
            z = self.subgraph_gate[2 * layer](z)
            x = self.subgraph_gate[2 * layer + 1](z) * x
            x = tf.concat((x, center_idx), axis=-1)

        if self.use_pooling_nn:
            #print('use_pooling_nn')
            for egde_pooling_layer in self.egde_pooling_nn:
                x = egde_pooling_layer(x)


        if self.subgraph_pooling == 'mean':
            #print("mean")
            x = tf.math.unsorted_segment_mean(x, node_to_subgraph2, num_segments=num)
        elif self.subgraph_pooling == 'add':
            #print('add')
            x = tf.math.unsorted_segment_sum(x, node_to_subgraph2, num_segments=num)
        elif self.subgraph_pooling == 'mean-context':
            #print('mean-context')
            x = tf.concat((tf.math.unsorted_segment_mean(x, node_to_subgraph2, num_segments=num),
                           x_node), axis=-1)

        if node_emb_only:
            return x

        if self.use_pooling_nn:
            x = self.node_pooling_nn(x)

        if aggr == 'mean':
            #print('aggr = mean')
            return tf.math.unsorted_segment_mean(x, node_to_subgraph2, num_segments=num)
        elif aggr == 'add':
            #print('aggr = add')
            return tf.math.unsorted_segment_sum(x, node_to_subgraph2, num_segments=num)



    def build(self, G, x, edge_index, z, node_to_subgraph2, node_to_original_node, center_index, rd, new_z2):
        print("I2GCN------------------------------------------------")

        self.node_to_subgraph2 = node_to_subgraph2
        self.node_to_original_node = node_to_original_node
        self.center_index = center_index
        self.rd = tf.cast(np.array(rd), tf.float32)

        #node embedding
        self.x = x

        for elem in enumerate(self.x):
            for elem_in in enumerate(elem):
                if elem_in[0] == 1:
                    if self.new_x == []:
                        self.new_x = elem_in[1]
                    else:
                        self.new_x = tf.concat([self.new_x, elem_in[1]], axis=0)

        row = np.shape(z)[0]

        #concatenate with continuous node features
        self.new_x =  tf.reshape(self.new_x, (-1, self.in_features))
        self.new_x_2 = self.fnn_rew_x_2.build(self.new_x)
        self.new_x = tf.concat((self.new_x, self.new_x_2), axis=-1)                    #Tensor("concat_2:0", shape=(615, 9), dtype=float32)


        for layer  in range(self.num_layers):
            z_emb = tf.cast(np.array(z), tf.float32)
            rd = tf.cast(self.rd, tf.float32)
            z_emb = self.z_embedding_list[layer](z_emb)
            if self.use_rd:
                z_emb = z_emb + self.rd_projection_list[layer](rd)

            self.new_x = tf.concat((self.new_x, z_emb), axis=-1)

            input_size = int(np.shape(self.new_x)[1])
            output_size = int(np.shape(self.new_x)[0])

            #new_z2 = tf.cast(np.array(new_z2), tf.float32)
            new_z2 = tf.cast(new_z2, tf.float32)

            self.new_x = GCN(inp=input_size, hid=(input_size, 64),
                             out=(64, 64), dropout=0.7).build(self.new_x, new_z2)

            if self.double_pooling:
                x = self.graph_pooling(self.new_x, z_emb, layer, node_emb_only=True,
                                          node_to_original_node=self.node_to_original_node,
                                          node_to_subgraph2=self.node_to_subgraph2, num=row, center_idx=self.center_index)
                x = self.double_nns[3 * layer](x)
                x = self.double_nns[3 * layer + 1](x)
                x = self.double_nns[3 * layer + 2](x)

            x = self.norms[layer](x)

            if layer < len(self.convs) -1:
                x = tf.nn.elu(x)

            if layer >0 and self.res:
                x = x+x0
            x0 =x

        # graph pooling
        # distance embedding
        ########################################################################
        # z_emb = tf.cast(np.array(z), tf.float32)
        # rd = tf.cast(self.rd, tf.float32)
        # #z_emb = self.z_embedding_list[-1](z_emb)
        # z_emb = tf.keras.layers.Dense(64)(z_emb)
        # if self.use_rd:
        #     z_emb = z_emb + self.rd_projection_list[-1](rd)
        #
        # x = self.graph_pooling(x, z_emb, len(self.convs), node_emb_only=True,
        #                        node_to_original_node=self.node_to_original_node,
        #                        node_to_subgraph2=self.node_to_subgraph2, num=row, center_idx=self.center_index)
        ###########################################################################

        x = tf.nn.elu(self.fc1(x))
        x = tf.nn.elu(self.fc2(x))
        x = self.fc3(x)

        return x

class I2GAT(object):
    def __init__(self,in_features, embedding_size_deg, num_layers=5, subgraph_pooling='mean', subgraph2_pooling='mean',
                 use_pooling_nn=False,use_rd=False,double_pooling=False, gate=False):
        assert (subgraph_pooling=='mean' or subgraph_pooling=='add' or subgraph_pooling=='mean-context') \
               and (subgraph2_pooling=='mean' or subgraph2_pooling=='add' or subgraph2_pooling=='center'
                    or subgraph2_pooling=='mean-center' or subgraph2_pooling=='mean-center-side')
        self.in_features = in_features
        self.num_layers = num_layers
        self.subgraph_pooling = subgraph_pooling
        self.subgraph2_pooling =subgraph2_pooling
        s2_dim = 2 if subgraph2_pooling=='mean-center' else 3 if subgraph2_pooling=='mean-center-side' else 1
        s2_dim = s2_dim + 1 if subgraph_pooling == 'mean-context' else s2_dim
        self.use_rd = use_rd
        if self.use_rd:
            self.rd_projection = tf.keras.layers.Dense(8)

        self.node_type_embedding = SingleLayerFNN(inp_size=8, inp_shape=(100,8), name='positional_awareness')
        self.double_pooling = double_pooling
        self.res = True
        self.gate = gate
        if self.gate:
            self.subgraph_gate = []
        self.new_x = []
        self.embedding_size_deg = embedding_size_deg

        # node label feature
        self.convs = []
        self.norms = []
        self.z_embedding_list =  []
        if self.use_rd:
            self.rd_projection_list = []

        if self.double_pooling:
            self.double_nns = []
        M_in, M_out = self.embedding_size_deg+2, 64
        self.GINConv = SingleLayerFNN(inp_size=M_out, inp_shape=(2*M_in,M_out), name='positional_awareness')
        self.fnn_rew_x_2 = SingleLayerFNN(inp_size=2, inp_shape=(self.embedding_size_deg, 2), name='positional_awareness')


        #first layer  157
        #self.convs.append(SingleLayerFNN(inp_size=M_out, inp_shape=(2*M_in,M_out), name='positional_awareness'))
        self.convs.append(tf.keras.layers.Dense(M_out))
        self.z_embedding_list.append(tf.keras.layers.Dense(M_in))
        if self.use_rd:
            self.rd_projection_list.append( tf.keras.layers.Dense(M_in))
        if self.gate:
            self.subgraph_gate.append(tf.keras.layers.Dense(M_out))   #activation='sigmiod'
            self.subgraph_gate.append(tf.keras.layers.Activation('sigmoid'))
        self.norms.append(tf.keras.layers.BatchNormalization())
        if self.double_pooling:
            self.double_nns.append(tf.keras.layers.Dense(128))   #activation='relu'
            self.double_nns.append(tf.keras.layers.Activation('relu'))
            self.double_nns.append(tf.keras.layers.Dense(M_out))

        for i in range(num_layers-1):
            M_in, M_out = M_out, 64
            self.z_embedding_list.append(tf.keras.layers.Dense(M_in))
            if self.use_rd:
                self.rd_projection_list.append(tf.keras.layers.Dense(M_in))
            if self.gate:
                self.subgraph_gate.append(tf.keras.layers.Dense(M_out)) #activation='sigmiod'
                self.subgraph_gate.append(tf.keras.layers.Activation('sigmoid'))

            self.convs.append(tf.keras.layers.Dense(M_out))
            self.norms.append(tf.keras.layers.BatchNormalization())

            if self.double_pooling:
                self.double_nns.append(tf.keras.layers.Dense(128))  #activation='relu'
                self.double_nns.append(tf.keras.layers.Activation('relu'))
                self.double_nns.append(tf.keras.layers.Dense(M_out))

        # MLPs for hierarchical pooling
        self.egde_pooling_nn = []
        self.node_pooling_nn = []
        if use_pooling_nn:
            self.egde_pooling_nn.append(
                tf.keras.layers.Dense(s2_dim * M_out))  #activation='relu'
            self.egde_pooling_nn.append(
                tf.keras.layers.Activation('relu'))  # activation='relu'
            self.egde_pooling_nn.append(
                tf.keras.layers.Dense(s2_dim * M_out))

            self.node_pooling_nn.append(
                tf.keras.layers.Dense(s2_dim * M_out))  # activation='relu'
            self.node_pooling_nn.append(
                tf.keras.layers.Activation('relu'))  # activation='relu'
            self.node_pooling_nn.append(
                tf.keras.layers.Dense(s2_dim * M_out))
        self.use_pooling_nn = use_pooling_nn

        #final graph pooling
        if self.use_rd:
            self.rd_projection_list.append(tf.keras.layers.Dense(M_out))
        if self.gate:
            self.subgraph_gate.append(tf.keras.layers.Dense(M_out))#activation='sigmiod'
            self.subgraph_gate.append(tf.keras.layers.Activation('sigmoid'))

        self.fc1 = tf.keras.layers.Dense(32)
        self.fc2 = tf.keras.layers.Dense(16)
        self.fc3 = tf.keras.layers.Dense(self.embedding_size_deg)

        self.gat = GAT(nheads=2, in_features=2*M_in, out_features=M_out,
                       dropout=0.7, alpha=0.2, concat=True, dtype=tf.float32)

        self.gat0 = GAT(nheads=1, in_features=9, out_features=64, dropout=0.7, alpha=0.7, concat=True, dtype=tf.float32)

        self.gat1 = GAT(nheads=1, in_features=64, out_features=64, dropout=0.7, alpha=0.7, concat=True, dtype=tf.float32)


    def center_pool(self, x, indices, num_segments=None):
        global_feat = tf.reduce_mean(x, axis=0)
        center_feat = x - global_feat
        subgraph_feat = center_feat * tf.one_hot(64, 64)
        subgraph_feat = tf.math.unsorted_segment_sum(subgraph_feat, indices, num_segments)

        return subgraph_feat

    def graph_pooling(self, x, z, layer=None, aggr='mean', node_emb_only=False, node_to_original_node=None, node_to_subgraph2=None, num=None, center_idx=None):

        #print("graph_pooling")
        # functions = [func for func in dir(tf.keras.layers) if callable(getattr(tf.keras.layers, func))]
        # print(functions)
        if self.subgraph_pooling == 'mean-context':
            #print('mean-context')
            x_node = tf.math.unsorted_segment_mean(x, node_to_original_node, num_segments=num)

        if self.subgraph2_pooling == 'mean':
            #print("mean")
            if self.gate:
                z = self.subgraph_gate[2 * layer](z)
                x = self.subgraph_gate[2 * layer +1](z) * x
            x = tf.math.unsorted_segment_mean(x, node_to_subgraph2, num_segments=num)
        elif self.subgraph2_pooling == 'add':
            #print('add')
            x = tf.math.unsorted_segment_sum(x, node_to_subgraph2, num_segments=num)
        elif self.subgraph2_pooling == 'center':
            #print('center')
            x = self.center_pool(x, node_to_subgraph2, num_segments=num)
        elif self.subgraph2_pooling == 'mean-center':
            #print('mean-center')
            x = tf.concat((tf.math.unsorted_segment_mean(x, node_to_subgraph2, num_segments=num),
                           self.center_pool(x, node_to_subgraph2, num_segments=num)), axis=-1)
        elif self.subgraph2_pooling == 'mean-center-side':
            z = self.subgraph_gate[2 * layer](z)
            x = self.subgraph_gate[2 * layer + 1](z) * x
            x = tf.concat((x, center_idx), axis=-1)

        if self.use_pooling_nn:
            #print('use_pooling_nn')
            for egde_pooling_layer in self.egde_pooling_nn:
                x = egde_pooling_layer(x)


        if self.subgraph_pooling == 'mean':
            #print("mean")
            x = tf.math.unsorted_segment_mean(x, node_to_subgraph2, num_segments=num)
        elif self.subgraph_pooling == 'add':
            #print('add')
            x = tf.math.unsorted_segment_sum(x, node_to_subgraph2, num_segments=num)
        elif self.subgraph_pooling == 'mean-context':
            #print('mean-context')
            x = tf.concat((tf.math.unsorted_segment_mean(x, node_to_subgraph2, num_segments=num),
                           x_node), axis=-1)

        if node_emb_only:
            return x

        if self.use_pooling_nn:
            x = self.node_pooling_nn(x)

        if aggr == 'mean':
            #print('aggr = mean')
            return tf.math.unsorted_segment_mean(x, node_to_subgraph2, num_segments=num)
        elif aggr == 'add':
            #print('aggr = add')
            return tf.math.unsorted_segment_sum(x, node_to_subgraph2, num_segments=num)



    def build(self, G, x, edge_index, z, node_to_subgraph2, node_to_original_node, center_index, rd, new_z2):
        print("I2GAT------------------------------------------------")

        self.node_to_subgraph2 = node_to_subgraph2
        self.node_to_original_node = node_to_original_node
        self.center_index = center_index
        self.rd = tf.cast(np.array(rd), tf.float32)

        #node embedding
        self.x = x

        for elem in enumerate(self.x):
            for elem_in in enumerate(elem):
                if elem_in[0] == 1:
                    if self.new_x == []:
                        self.new_x = elem_in[1]
                    else:
                        self.new_x = tf.concat([self.new_x, elem_in[1]], axis=0)

        row = np.shape(z)[0]

        #concatenate with continuous node features
        self.new_x =  tf.reshape(self.new_x, (-1, self.in_features))
        self.new_x_2 = self.fnn_rew_x_2.build(self.new_x)
        self.new_x = tf.concat((self.new_x, self.new_x_2), axis=-1)                    #Tensor("concat_2:0", shape=(615, 9), dtype=float32)


        for layer  in range(self.num_layers):
            z_emb = tf.cast(np.array(z), tf.float32)
            rd = tf.cast(self.rd, tf.float32)
            z_emb =  self.z_embedding_list[layer](z_emb)
            if self.use_rd:
                z_emb = z_emb + self.rd_projection_list[layer](rd)

            self.new_x = tf.concat((self.new_x, z_emb), axis=-1)

            input_size = int(np.shape(self.new_x)[1])

            new_z2 = tf.cast(np.array(new_z2), tf.float32)

            self.new_x = GAT(nheads=1, in_features=input_size, out_features=64, dropout=0.7, alpha=0.7, concat=True, dtype=tf.float32).build(self.new_x, new_z2)

            self.new_z2_matrix = new_z2 + 2 * np.eye(int(np.shape(self.new_x)[0]))

            self.new_x = self.new_z2_matrix @ self.new_x


            if self.double_pooling:
                x = self.graph_pooling(self.new_x, z_emb, layer, node_emb_only=True,
                                          node_to_original_node=self.node_to_original_node,
                                          node_to_subgraph2=self.node_to_subgraph2, num=row, center_idx=self.center_index)
                x = self.double_nns[3 * layer](x)
                x = self.double_nns[3 * layer + 1](x)
                x = self.double_nns[3 * layer + 2](x)

            x = self.norms[layer](x)

            if layer < len(self.convs) -1:
                x = tf.nn.elu(x)

            if layer >0 and self.res:
                x = x+x0
            x0 =x

        # graph pooling
        # distance embedding
        ##########################################################################
        # z_emb = tf.cast(np.array(z), tf.float32)
        # rd = tf.cast(self.rd, tf.float32)
        # z_emb = tf.keras.layers.Dense(64)(z_emb)
        # if self.use_rd:
        #     z_emb = z_emb + self.rd_projection_list[-1](rd)
        #
        # x = self.graph_pooling(x, z_emb, len(self.convs), node_emb_only=True,
        #                        node_to_original_node=self.node_to_original_node,
        #                        node_to_subgraph2=self.node_to_subgraph2, num=row, center_idx=self.center_index)
        ##########################################################################

        x = tf.nn.elu(self.fc1(x))
        x = tf.nn.elu(self.fc2(x))
        x = self.fc3(x)

        return x

class I2GNN(object):
    def __init__(self,in_features, embedding_size_deg, num_layers=5, subgraph_pooling='mean', subgraph2_pooling='mean',
                 use_pooling_nn=False,use_rd=False,double_pooling=False, gate=False):
        assert (subgraph_pooling=='mean' or subgraph_pooling=='add' or subgraph_pooling=='mean-context') \
               and (subgraph2_pooling=='mean' or subgraph2_pooling=='add' or subgraph2_pooling=='center'
                    or subgraph2_pooling=='mean-center' or subgraph2_pooling=='mean-center-side')
        self.in_features = in_features
        self.num_layers = num_layers
        self.subgraph_pooling = subgraph_pooling
        self.subgraph2_pooling =subgraph2_pooling
        s2_dim = 2 if subgraph2_pooling=='mean-center' else 3 if subgraph2_pooling=='mean-center-side' else 1
        s2_dim = s2_dim + 1 if subgraph_pooling == 'mean-context' else s2_dim
        self.use_rd = use_rd
        if self.use_rd:
            self.rd_projection = tf.keras.layers.Dense(8)

        self.node_type_embedding = SingleLayerFNN(inp_size=8, inp_shape=(100,8), name='positional_awareness')
        self.double_pooling = double_pooling
        self.res = True
        self.gate = gate
        if self.gate:
            self.subgraph_gate = []
        self.new_x = []
        self.embedding_size_deg = embedding_size_deg

        # node label feature
        self.convs = []
        self.norms = []
        self.z_embedding_list =  []
        if self.use_rd:
            self.rd_projection_list = []

        if self.double_pooling:
            self.double_nns = []
        M_in, M_out = self.embedding_size_deg+2, 64
        self.GINConv = SingleLayerFNN(inp_size=M_out, inp_shape=(2*M_in,M_out), name='positional_awareness')
        self.fnn_rew_x_2 = SingleLayerFNN(inp_size=2, inp_shape=(self.embedding_size_deg, 2), name='positional_awareness')


        #first layer  157
        #self.convs.append(SingleLayerFNN(inp_size=M_out, inp_shape=(2*M_in,M_out), name='positional_awareness'))
        self.convs.append(tf.keras.layers.Dense(M_out))
        self.z_embedding_list.append(tf.keras.layers.Dense(M_in))
        if self.use_rd:
            self.rd_projection_list.append( tf.keras.layers.Dense(M_in))
        if self.gate:
            self.subgraph_gate.append(tf.keras.layers.Dense(M_out))   #activation='sigmiod'
            self.subgraph_gate.append(tf.keras.layers.Activation('sigmoid'))
        self.norms.append(tf.keras.layers.BatchNormalization())
        if self.double_pooling:
            self.double_nns.append(tf.keras.layers.Dense(128))   #activation='relu'
            self.double_nns.append(tf.keras.layers.Activation('relu'))
            self.double_nns.append(tf.keras.layers.Dense(M_out))

        for i in range(num_layers-1):
            M_in, M_out = M_out, 64
            self.z_embedding_list.append(tf.keras.layers.Dense(M_in))
            if self.use_rd:
                self.rd_projection_list.append(tf.keras.layers.Dense(M_in))
            if self.gate:
                self.subgraph_gate.append(tf.keras.layers.Dense(M_out)) #activation='sigmiod'
                self.subgraph_gate.append(tf.keras.layers.Activation('sigmoid'))

            self.convs.append(tf.keras.layers.Dense(M_out))
            self.norms.append(tf.keras.layers.BatchNormalization())

            if self.double_pooling:
                self.double_nns.append(tf.keras.layers.Dense(128))  #activation='relu'
                self.double_nns.append(tf.keras.layers.Activation('relu'))
                self.double_nns.append(tf.keras.layers.Dense(M_out))

        # MLPs for hierarchical pooling
        self.egde_pooling_nn = []
        self.node_pooling_nn = []
        if use_pooling_nn:
            self.egde_pooling_nn.append(
                tf.keras.layers.Dense(s2_dim * M_out))  #activation='relu'
            self.egde_pooling_nn.append(
                tf.keras.layers.Activation('relu'))  # activation='relu'
            self.egde_pooling_nn.append(
                tf.keras.layers.Dense(s2_dim * M_out))

            self.node_pooling_nn.append(
                tf.keras.layers.Dense(s2_dim * M_out))  # activation='relu'
            self.node_pooling_nn.append(
                tf.keras.layers.Activation('relu'))  # activation='relu'
            self.node_pooling_nn.append(
                tf.keras.layers.Dense(s2_dim * M_out))
        self.use_pooling_nn = use_pooling_nn

        #final graph pooling
        if self.use_rd:
            self.rd_projection_list.append(tf.keras.layers.Dense(M_out))
        if self.gate:
            self.subgraph_gate.append(tf.keras.layers.Dense(M_out))#activation='sigmiod'
            self.subgraph_gate.append(tf.keras.layers.Activation('sigmoid'))

        self.fc1 = tf.keras.layers.Dense(32)
        self.fc2 = tf.keras.layers.Dense(16)
        self.fc3 = tf.keras.layers.Dense(self.embedding_size_deg)

    def center_pool(self, x, indices, num_segments=None):
        global_feat = tf.reduce_mean(x, axis=0)
        center_feat = x - global_feat
        subgraph_feat = center_feat * tf.one_hot(64, 64)
        subgraph_feat = tf.math.unsorted_segment_sum(subgraph_feat, indices, num_segments)

        return subgraph_feat

    def graph_pooling(self, x, z, layer=None, aggr='mean', node_emb_only=False, node_to_original_node=None, node_to_subgraph2=None, num=None, center_idx=None):

        #print("graph_pooling")
        # functions = [func for func in dir(tf.keras.layers) if callable(getattr(tf.keras.layers, func))]
        # print(functions)
        if self.subgraph_pooling == 'mean-context':
            #print('mean-context')
            x_node = tf.math.unsorted_segment_mean(x, node_to_original_node, num_segments=num)

        if self.subgraph2_pooling == 'mean':
            #print("mean")
            if self.gate:
                z = self.subgraph_gate[2 * layer](z)
                x = self.subgraph_gate[2 * layer +1](z) * x
            x = tf.math.unsorted_segment_mean(x, node_to_subgraph2, num_segments=num)
        elif self.subgraph2_pooling == 'add':
            #print('add')
            x = tf.math.unsorted_segment_sum(x, node_to_subgraph2, num_segments=num)
        elif self.subgraph2_pooling == 'center':
            #print('center')
            x = self.center_pool(x, node_to_subgraph2, num_segments=num)
        elif self.subgraph2_pooling == 'mean-center':
            #print('mean-center')
            x = tf.concat((tf.math.unsorted_segment_mean(x, node_to_subgraph2, num_segments=num),
                           self.center_pool(x, node_to_subgraph2, num_segments=num)), axis=-1)
        elif self.subgraph2_pooling == 'mean-center-side':
            z = self.subgraph_gate[2 * layer](z)
            x = self.subgraph_gate[2 * layer + 1](z) * x
            x = tf.concat((x, center_idx), axis=-1)

        if self.use_pooling_nn:
            #print('use_pooling_nn')
            for egde_pooling_layer in self.egde_pooling_nn:
                x = egde_pooling_layer(x)


        if self.subgraph_pooling == 'mean':
            #print("mean")
            x = tf.math.unsorted_segment_mean(x, node_to_subgraph2, num_segments=num)
        elif self.subgraph_pooling == 'add':
            #print('add')
            x = tf.math.unsorted_segment_sum(x, node_to_subgraph2, num_segments=num)
        elif self.subgraph_pooling == 'mean-context':
            #print('mean-context')
            x = tf.concat((tf.math.unsorted_segment_mean(x, node_to_subgraph2, num_segments=num),
                           x_node), axis=-1)

        if node_emb_only:
            return x

        if self.use_pooling_nn:
            x = self.node_pooling_nn(x)

        if aggr == 'mean':
            #print('aggr = mean')
            return tf.math.unsorted_segment_mean(x, node_to_subgraph2, num_segments=num)
        elif aggr == 'add':
            #print('aggr = add')
            return tf.math.unsorted_segment_sum(x, node_to_subgraph2, num_segments=num)



    def build(self, G, x, edge_index, z, node_to_subgraph2, node_to_original_node, center_index, rd, new_z2):
        print("I2GNN------------------------------------------------")

        self.node_to_subgraph2 = node_to_subgraph2
        self.node_to_original_node = node_to_original_node
        self.center_index = center_index
        self.rd = tf.cast(np.array(rd), tf.float32)
        #self.rd = tf.expand_dims(self.rd, axis=1)

        #node embedding
        self.x = x

        for elem in enumerate(self.x):
            for elem_in in enumerate(elem):
                if elem_in[0] == 1:
                    if self.new_x == []:
                        self.new_x = elem_in[1]
                    else:
                        self.new_x = tf.concat([self.new_x, elem_in[1]], axis=0)


        # self.x_matrix = []
        # for i in enumerate(self.x):
        #     self.x_matrix.append(i)
        #
        #
        # edge_index = np.transpose(edge_index)
        row = np.shape(z)[0]
        #
        # self.new_x = []
        #
        # for i in range(row):
        #     start = edge_index[i][0]
        #     end = edge_index[i][1]
        #     start_emb = self.x[self.x_matrix[start][1]]
        #     end_emb = self.x[self.x_matrix[end][1]]
        #     emb = start_emb+end_emb
        #     self.new_x.append(emb)



        #concatenate with continuous node features
        self.new_x =  tf.reshape(self.new_x, (-1, self.in_features))
        #self.new_x_2 = tf.reduce_sum(self.new_x, axis=-1, keepdims=True)
        self.new_x_2 = self.fnn_rew_x_2.build(self.new_x)
        self.new_x = tf.concat((self.new_x, self.new_x_2), axis=-1)                    #Tensor("concat_2:0", shape=(615, 9), dtype=float32)

        layer = 0
        for conv in self.convs:
            #print(layer)
            z_emb = tf.cast(np.array(z), tf.float32)
            #z_emb = tf.expand_dims(z_emb, axis=1)

            z_emb = self.z_embedding_list[layer](z_emb)
            if self.use_rd:
                z_emb = z_emb + self.rd_projection_list[layer](self.rd)

            self.new_x = tf.concat((self.new_x, z_emb), axis=-1)
            self.new_x = conv(self.new_x)

            if self.double_pooling:
                x = self.graph_pooling(self.new_x, z_emb, layer, node_emb_only=True,
                                          node_to_original_node=self.node_to_original_node,
                                          node_to_subgraph2=self.node_to_subgraph2, num=row, center_idx=self.center_index)
                x = self.double_nns[3 * layer](x)
                x = self.double_nns[3 * layer + 1](x)
                x = self.double_nns[3 * layer + 2](x)

            x = self.norms[layer](x)

            if layer < len(self.convs) -1:
                x = tf.nn.elu(x)

            if layer >0 and self.res:
                x = x+x0
            x0 =x
            layer += 1

        # graph pooling
        # distance embedding
        ##################################################################
        # z_emb = tf.cast(np.array(z)[:,0], tf.float32)
        # z_emb = tf.expand_dims(z_emb, axis=1)
        # z_emb = tf.keras.layers.Dense(64)(z_emb)
        # if self.use_rd:
        #     z_emb = z_emb + self.rd_projection_list[-1](self.rd)
        #
        # x = self.graph_pooling(x, z_emb, len(self.convs), node_emb_only=True,
        #                        node_to_original_node=self.node_to_original_node,
        #                        node_to_subgraph2=self.node_to_subgraph2, num=row, center_idx=self.center_index)
        ##################################################################

        x = tf.nn.elu(self.fc1(x))
        x = tf.nn.elu(self.fc2(x))
        x = self.fc3(x)

        #layer += 1
        return x

class I2TF(object):
    def __init__(self,in_features, embedding_size_deg, num_layers=5, subgraph_pooling='mean', subgraph2_pooling='mean',
                 use_pooling_nn=False,use_rd=False,double_pooling=False, gate=False):
        assert (subgraph_pooling=='mean' or subgraph_pooling=='add' or subgraph_pooling=='mean-context') \
               and (subgraph2_pooling=='mean' or subgraph2_pooling=='add' or subgraph2_pooling=='center'
                    or subgraph2_pooling=='mean-center' or subgraph2_pooling=='mean-center-side')
        self.in_features = in_features
        self.num_layers = num_layers
        self.subgraph_pooling = subgraph_pooling
        self.subgraph2_pooling =subgraph2_pooling
        s2_dim = 2 if subgraph2_pooling=='mean-center' else 3 if subgraph2_pooling=='mean-center-side' else 1
        s2_dim = s2_dim + 1 if subgraph_pooling == 'mean-context' else s2_dim
        self.use_rd = use_rd
        if self.use_rd:
            self.rd_projection = tf.keras.layers.Dense(8)

        self.node_type_embedding = SingleLayerFNN(inp_size=8, inp_shape=(100,8), name='positional_awareness')
        self.double_pooling = double_pooling
        self.res = True
        self.gate = gate
        if self.gate:
            self.subgraph_gate = []
        self.new_x = []
        self.embedding_size_deg = embedding_size_deg

        # node label feature
        self.convs = []
        self.norms = []
        self.z_embedding_list =  []
        if self.use_rd:
            self.rd_projection_list = []

        if self.double_pooling:
            self.double_nns = []
        M_in, M_out = self.embedding_size_deg+2, 64
        self.GINConv = SingleLayerFNN(inp_size=M_out, inp_shape=(2*M_in,M_out), name='positional_awareness')
        self.fnn_rew_x_2 = SingleLayerFNN(inp_size=2, inp_shape=(self.embedding_size_deg, 2), name='positional_awareness')


        #first layer  157
        #self.convs.append(SingleLayerFNN(inp_size=M_out, inp_shape=(2*M_in,M_out), name='positional_awareness'))
        self.convs.append(tf.keras.layers.Dense(M_out))
        self.z_embedding_list.append(tf.keras.layers.Dense(M_in))
        if self.use_rd:
            self.rd_projection_list.append( tf.keras.layers.Dense(M_in))
        if self.gate:
            self.subgraph_gate.append(tf.keras.layers.Dense(M_out))   #activation='sigmiod'
            self.subgraph_gate.append(tf.keras.layers.Activation('sigmoid'))
        self.norms.append(tf.keras.layers.BatchNormalization())
        if self.double_pooling:
            self.double_nns.append(tf.keras.layers.Dense(128))   #activation='relu'
            self.double_nns.append(tf.keras.layers.Activation('relu'))
            self.double_nns.append(tf.keras.layers.Dense(M_out))

        for i in range(num_layers-1):
            M_in, M_out = M_out, 64
            self.z_embedding_list.append(tf.keras.layers.Dense(M_in))
            if self.use_rd:
                self.rd_projection_list.append(tf.keras.layers.Dense(M_in))
            if self.gate:
                self.subgraph_gate.append(tf.keras.layers.Dense(M_out)) #activation='sigmiod'
                self.subgraph_gate.append(tf.keras.layers.Activation('sigmoid'))

            self.convs.append(tf.keras.layers.Dense(M_out))
            self.norms.append(tf.keras.layers.BatchNormalization())

            if self.double_pooling:
                self.double_nns.append(tf.keras.layers.Dense(128))  #activation='relu'
                self.double_nns.append(tf.keras.layers.Activation('relu'))
                self.double_nns.append(tf.keras.layers.Dense(M_out))

        # MLPs for hierarchical pooling
        self.egde_pooling_nn = []
        self.node_pooling_nn = []
        if use_pooling_nn:
            self.egde_pooling_nn.append(
                tf.keras.layers.Dense(s2_dim * M_out))  #activation='relu'
            self.egde_pooling_nn.append(
                tf.keras.layers.Activation('relu'))  # activation='relu'
            self.egde_pooling_nn.append(
                tf.keras.layers.Dense(s2_dim * M_out))

            self.node_pooling_nn.append(
                tf.keras.layers.Dense(s2_dim * M_out))  # activation='relu'
            self.node_pooling_nn.append(
                tf.keras.layers.Activation('relu'))  # activation='relu'
            self.node_pooling_nn.append(
                tf.keras.layers.Dense(s2_dim * M_out))
        self.use_pooling_nn = use_pooling_nn

        #final graph pooling
        if self.use_rd:
            self.rd_projection_list.append(tf.keras.layers.Dense(M_out))
        if self.gate:
            self.subgraph_gate.append(tf.keras.layers.Dense(M_out))#activation='sigmiod'
            self.subgraph_gate.append(tf.keras.layers.Activation('sigmoid'))

        self.fc1 = tf.keras.layers.Dense(32)
        self.fc2 = tf.keras.layers.Dense(16)
        self.fc3 = tf.keras.layers.Dense(self.embedding_size_deg)

        self.gat = GAT(nheads=2, in_features=2*M_in, out_features=M_out,
                       dropout=0.7, alpha=0.2, concat=True, dtype=tf.float32)

        self.gat0 = GAT(nheads=1, in_features=9, out_features=64, dropout=0.7, alpha=0.7, concat=True, dtype=tf.float32)

        self.gat1 = GAT(nheads=1, in_features=64, out_features=64, dropout=0.7, alpha=0.7, concat=True, dtype=tf.float32)


    def center_pool(self, x, indices, num_segments=None):
        global_feat = tf.reduce_mean(x, axis=0)
        center_feat = x - global_feat
        subgraph_feat = center_feat * tf.one_hot(64, 64)
        subgraph_feat = tf.math.unsorted_segment_sum(subgraph_feat, indices, num_segments)

        return subgraph_feat

    def graph_pooling(self, x, z, layer=None, aggr='mean', node_emb_only=False, node_to_original_node=None, node_to_subgraph2=None, num=None, center_idx=None):

        #print("graph_pooling")
        # functions = [func for func in dir(tf.keras.layers) if callable(getattr(tf.keras.layers, func))]
        # print(functions)
        if self.subgraph_pooling == 'mean-context':
            #print('mean-context')
            x_node = tf.math.unsorted_segment_mean(x, node_to_original_node, num_segments=num)

        if self.subgraph2_pooling == 'mean':
            #print("mean")
            if self.gate:
                z = self.subgraph_gate[2 * layer](z)
                x = self.subgraph_gate[2 * layer +1](z) * x
            x = tf.math.unsorted_segment_mean(x, node_to_subgraph2, num_segments=num)
        elif self.subgraph2_pooling == 'add':
            #print('add')
            x = tf.math.unsorted_segment_sum(x, node_to_subgraph2, num_segments=num)
        elif self.subgraph2_pooling == 'center':
            #print('center')
            x = self.center_pool(x, node_to_subgraph2, num_segments=num)
        elif self.subgraph2_pooling == 'mean-center':
            #print('mean-center')
            x = tf.concat((tf.math.unsorted_segment_mean(x, node_to_subgraph2, num_segments=num),
                           self.center_pool(x, node_to_subgraph2, num_segments=num)), axis=-1)
        elif self.subgraph2_pooling == 'mean-center-side':
            z = self.subgraph_gate[2 * layer](z)
            x = self.subgraph_gate[2 * layer + 1](z) * x
            x = tf.concat((x, center_idx), axis=-1)

        if self.use_pooling_nn:
            #print('use_pooling_nn')
            for egde_pooling_layer in self.egde_pooling_nn:
                x = egde_pooling_layer(x)


        if self.subgraph_pooling == 'mean':
            #print("mean")
            x = tf.math.unsorted_segment_mean(x, node_to_subgraph2, num_segments=num)
        elif self.subgraph_pooling == 'add':
            #print('add')
            x = tf.math.unsorted_segment_sum(x, node_to_subgraph2, num_segments=num)
        elif self.subgraph_pooling == 'mean-context':
            #print('mean-context')
            x = tf.concat((tf.math.unsorted_segment_mean(x, node_to_subgraph2, num_segments=num),
                           x_node), axis=-1)

        if node_emb_only:
            return x

        if self.use_pooling_nn:
            x = self.node_pooling_nn(x)

        if aggr == 'mean':
            #print('aggr = mean')
            return tf.math.unsorted_segment_mean(x, node_to_subgraph2, num_segments=num)
        elif aggr == 'add':
            #print('aggr = add')
            return tf.math.unsorted_segment_sum(x, node_to_subgraph2, num_segments=num)



    def build(self, G, x, edge_index, z, node_to_subgraph2, node_to_original_node, center_index, rd, new_z2):
        print("I2TF------------------------------------------------")

        self.node_to_subgraph2 = node_to_subgraph2
        self.node_to_original_node = node_to_original_node
        self.center_index = center_index
        self.rd = tf.cast(np.array(rd), tf.float32)

        #node embedding
        self.x = x

        for elem in enumerate(self.x):
            for elem_in in enumerate(elem):
                if elem_in[0] == 1:
                    if self.new_x == []:
                        self.new_x = elem_in[1]
                    else:
                        self.new_x = tf.concat([self.new_x, elem_in[1]], axis=0)

        row = np.shape(z)[0]

        # ptb  
        # self.new_x =  tf.reshape(self.new_x, (-1, self.in_features))
        # self.new_x_2 = self.fnn_rew_x_2.build(self.new_x)
        # self.new_x = tf.concat((self.new_x, self.new_x_2), axis=-1)
        x0 = None

        for layer  in range(self.num_layers):
            z_emb = tf.cast(np.array(z), tf.float32)
            rd = tf.cast(self.rd, tf.float32)
            z_emb = self.z_embedding_list[layer](z_emb)
            if self.use_rd:
                z_emb = z_emb + self.rd_projection_list[layer](rd)
                # z_emb = self.rd_projection_list[layer](rd)
                # z_emb = z_emb

            # self.new_x = tf.concat((self.new_x, z_emb), axis=-1)

            input_size = int(np.shape(self.new_x)[1])

            # new_z2 = tf.cast(np.array(new_z2), tf.float32)

            # new_z2 = new_z2 + 2 * np.eye(int(np.shape(self.new_x)[0]))    #cifar

            # self.new_x = GAT(nheads=1, in_features=input_size, out_features=64, dropout=0.7, alpha=0.7, concat=True, dtype=tf.float32).build(self.new_x, new_z2)

            self.transformer = TransformerEncoderLayer(input_size, nhead=1, dim_feedforward=512, dropout=0.3,
                activation="relu", batch_norm=True, pre_norm=False,
                gnn_type="gcn", se="gnn", k_hop=2,embedding_size_deg=input_size)

            self.new1_x = self.transformer.build(self.new_x, None, new_z2,
            subgraph_node_index=None, subgraph_edge_index=None,
            subgraph_edge_attr=None,
            subgraph_indicator_index=None,
            edge_attr=None, degree=None, ptr=None,
            return_attn=False,
            )


            self.new2_x = GCN(inp=input_size, hid=(input_size, input_size),
                           out=(input_size, input_size), dropout=0.1).build(self.new_x, tf.cast(new_z2, tf.float32))

            # self.gat = GAT(nheads=1, in_features=input_size, out_features=input_size, dropout=0.1, alpha=0.7, concat=True,
            #                  dtype=tf.float32).build(self.new_x, new_z2)
            # self.new2_x = self.gat.build(self.self_transform, self.Adjacency_matrix)



            # self.dgnn = DGNN(int(np.shape(self.new_x)[0]), 1, input_size, 1)
            # self.new_x = self.dgnn.build(self.new_x, new_z2)


            norm = RMSNorm(input_size)
            self.new1_x = norm(self.new1_x)
            self.new2_x = norm(self.new2_x)

            self.new1_x = tf.nn.dropout(self.new1_x, 0.1)
            self.new2_x = tf.nn.dropout(self.new2_x, 0.1)

            #################    NMT    CIFAR10
            self.new_x = 0.2 * self.new2_x + 0.8 * self.new1_x
            # #################    PTB
            # self.self_transform = tf.concat([self.new2_x, self.new1_x], axis=1)


            # self.new_x = self.fc3(self.new_x)


            if layer == 0:
                x0 = self.new_x
            else:
                self.new_x = (x0 + self.new_x)/2

        # self.new_x = tf.nn.dropout(self.new_x, keep_prob=0.1)
        # self.new_x = self.fc3(self.new_x)

        return self.new_x


'''
  Combines a bunch of embeddings together at a specific node
  g(sum_i(f_i)) = relu_g(Sum_i(relu_f(e_i* M_f + b_f))* M_g + b_g)
  g(sum_i(f_i)) = relu_g((Mask* (relu_f(E* M_f + b_f)))* M_g + b_g)
  To be more specific when the number of embeddings to combine is variable,
  we use a mask Ma
  Dimensions:
    E: N x d        placeholder
    M_f: d x d1     Variable
    b_f: 1 x d1     Variable
    Ma: 1 x N       placeholder
    M_g: d1 x d2    Variable
    b_g: 1 x d2     Variable
'''


class Aggregator(object):
    # N is the max number of children to be aggregated
    # d is the degree of embeddings
    # d1 is the degree of embedding transformation
    # d2 is degree of aggregation
    def __init__(self, d, d1=None, d2=None, use_mask=True, normalize_aggs=False,
                 small_nn=False, dtype=tf.float32):
        self.d = d
        self.d1 = d1
        self.d2 = d2
        self.normalize_aggs = normalize_aggs
        self.dtype = dtype

        if d1 is None:
            d1 = self.d1 = d
        if d2 is None:
            d2 = self.d2 = d

        self.use_mask = use_mask
        if use_mask:
            self.Ma = tf.placeholder(dtype, shape=(None, None))

        if small_nn:
            hidden_layer_f, hidden_layer_g = [], []
        else:
            hidden_layer_f = [self.d]
            hidden_layer_g = [self.d1]

        self.f = FNN(self.d, hidden_layer_f, self.d1, 'f', dtype=dtype)  ################################
        self.g = FNN(self.d1, hidden_layer_g, self.d2, 'g', dtype=dtype)

    def build(self, E, debug=False, mask=None):
        summ = 100

        f = tf.nn.relu(self.f.build(E))

        self.f_out = f

        if debug:
            f = tf.Print(f, [f], message='output of f: ', summarize=summ)

        if self.use_mask or mask is not None:
            if mask is None:
                mask = self.Ma

            g = tf.matmul(mask, f)
            if self.normalize_aggs:
                d = tf.cond(
                    tf.reduce_sum(mask) > 0,
                    lambda: tf.reduce_sum(mask),
                    lambda: 1.)

                g /= d

            if debug:
                print(f, g, self.Ma)
        else:
            g = tf.reduce_sum(f, 0, keepdims=True)

        if debug:
            g = tf.Print(g, [g], message='after mask: ', summarize=summ)

        g = tf.nn.relu(self.g.build(g))

        if debug:
            g = tf.Print(g, [g], message='output of g: ', summarize=summ)

        return g

    def get_ph(self):
        return self.Ma


class Classifier(object):
    def __init__(self, inp_size, hidden_layer_sizes, out_size, dtype=tf.float32):
        self.nn = FNN(inp_size, hidden_layer_sizes,
                      out_size, 'classifier', dtype=dtype)

    def build(self, inp_tensor):
        return self.nn.build(inp_tensor)


class SAGEMessenger(object):
    """
    Implementation of GraphSAGE-like algorithm for embedding to be used in the RL policy.

    Paper: "Inductive Representation Learning on Large Graphs" (https://arxiv.org/pdf/1706.02216.pdf)

    Parameters:

    - `embedding_size` - int - degree of embeddings
    - `embedding_transformation_deg` - int
    - `small_nn` - currently not used
    - `sample_ratio` - float [0,1] - what part of a node's neighbours are used to calculate its embeddings
    - `hops` - int [1,2] - how many hops away need to be aggregated
    - `aggregation` - {'mean', 'max', 'min', 'sum'} - how are a node's neighbours aggregated. Default is 'mean'
    """
    def __init__(self, embedding_size_deg,
                 embedding_transformation_deg,
                 small_nn=False,
                 sage_sample_ratio=0.5,
                 sage_hops=2,
                 sage_aggregation='max',
                 sage_dropout_rate=0.5,
                 position_aware=True,
                 single_layer_perceptron=False,
                 pgnn_c=0.5,
                 pgnn_neigh_cutoff=6,
                 pgnn_anchor_exponent=1,
                 pgnn_aggregation='max',
                 dtype=tf.float32,
                 embs=(0,1,2),
                 embs_combine_mode='add',
                 position = 1
                 ):

        self.embedding_size_deg = embedding_size_deg
        self.embedding_transformation_deg = embedding_transformation_deg
        self.small_nn = small_nn

        self.hops = sage_hops
        self.aggregation = sage_aggregation
        self.dropout_rate = sage_dropout_rate
        self.sample_ratio = sage_sample_ratio

        self.position_aware = position_aware
        self.single_layer_perceptron = single_layer_perceptron

        self.pgnn_c = pgnn_c
        self.pgnn_neigh_cutoff = pgnn_neigh_cutoff
        self.pgnn_anchor_exponent = pgnn_anchor_exponent
        self.pgnn_aggregation = pgnn_aggregation

        self.memo = {}

        self.samples = {}
        self.anchor_sets = []
        self.fnns = {}
        self.distances = {}
        self.distances_numpy = []
        self.nodes = []

        self._init_fnns()

        self.sum_node = 0
        self.sum_neighbor = 0

        self.node_index = []
        self.neighbor_index = []
        #self.SubgraphGNNKernel = SubgraphGNNKernel
        self.subgraph_x = []
        self.embs = embs
        self.embs_combine_mode = embs_combine_mode
        self.position = position

        self.n_cross = 4
        self.n_self = 4
        self.num_heads = 2

        self.num_hop = 8
        self.cross_hop = 8
        self.self_hop = 8

        self.type = 'SELF'  # 'CROSS','SELF', 'ADD'


        # self.base_gcn = GraphConvSparse( self.embedding_size_deg+self.embedding_size_deg, self.embedding_size_deg)
        self.base_gcn = SingleLayerFNN(inp_size=self.embedding_size_deg, inp_shape=(self.embedding_size_deg, self.embedding_size_deg),name='base_gcn')


        self.base_gcn_node = SingleLayerFNN(inp_size=self.embedding_size_deg,
                                           inp_shape=(
                                               self.embedding_size_deg+self.embedding_size_deg, self.embedding_size_deg),
                                           name='positional_awareness')

        self.base_gcn_pos = SingleLayerFNN(inp_size=self.embedding_size_deg,
                                           inp_shape=(
                                               self.embedding_size_deg + self.embedding_size_deg, self.embedding_size_deg),
                                           name='positional_awareness')

        self.gcn = GCN(inp=self.embedding_size_deg, hid=(self.embedding_size_deg, self.n_cross*self.embedding_size_deg),
                       out=(self.embedding_size_deg, self.embedding_size_deg), dropout=self.dropout_rate)

        self.gcn_cross = GCN(inp=self.embedding_size_deg, hid=(self.embedding_size_deg, self.n_cross*self.embedding_size_deg),
                       out=(self.embedding_size_deg, self.embedding_size_deg), dropout=self.dropout_rate)

        self.gcn_self = GCN(inp=self.embedding_size_deg, hid=(self.embedding_size_deg, self.n_self* self.embedding_size_deg),
                       out=(self.embedding_size_deg, self.embedding_size_deg), dropout=self.dropout_rate)

        #self.GraphAttentionLayer = GraphAttentionLayer(in_features=self.embedding_size_deg, out_features=self.embedding_size_deg, dropout=self.dropout_rate, alpha=0.2, concat=True, dtype=tf.float32)

        self.gat = GAT(nheads=1, in_features=self.embedding_size_deg, out_features=2*self.embedding_size_deg,
                       dropout=self.dropout_rate, alpha=0.2, type=None, concat=True, dtype=tf.float32)

        self.i2gnn = I2GNN(in_features=self.embedding_size_deg, embedding_size_deg=self.embedding_size_deg,
                           num_layers=1, subgraph_pooling='mean', subgraph2_pooling='mean', use_pooling_nn=True,
                           use_rd=True, double_pooling=True, gate=True)

        self.i2gat = I2GAT(in_features=self.embedding_size_deg, embedding_size_deg=self.embedding_size_deg,
                           num_layers=1, subgraph_pooling='mean', subgraph2_pooling='mean', use_pooling_nn=True,
                           use_rd=True, double_pooling=True, gate=True)

        self.i2TF = I2TF(in_features=self.embedding_size_deg, embedding_size_deg=self.embedding_size_deg,
                           num_layers=1, subgraph_pooling='mean', subgraph2_pooling='mean', use_pooling_nn=True,
                           use_rd=True, double_pooling=True, gate=True)

        self.i2gcn = I2GCN(in_features=self.embedding_size_deg, embedding_size_deg=self.embedding_size_deg,
                           num_layers=1, subgraph_pooling='mean', subgraph2_pooling='mean', use_pooling_nn=True,
                           use_rd=True, double_pooling=True, gate=True)

        self.cross_attention = CrossAttention(self.n_cross * self.embedding_size_deg, num_heads=self.num_heads)

        self.learn = LinearAttention(self.n_self * self.embedding_size_deg, num_heads=self.num_heads)

        self.fnn = FNN(hidden_layer_sizes=[self.embedding_size_deg],
                                  inp_size=self.embedding_size_deg,
                                  out_size=self.embedding_transformation_deg,
                                  name='self_transform')

        self.fnn_out_cross = FNN(hidden_layer_sizes=[self.n_cross*self.embedding_size_deg],
                                  inp_size=self.n_cross*self.embedding_size_deg,
                                  out_size=self.embedding_transformation_deg,
                                  name='self_transform')

        self.fnn_out_self = FNN(hidden_layer_sizes=[self.n_self*self.embedding_size_deg],
                                  inp_size=self.n_self*self.embedding_size_deg,
                                  out_size=self.embedding_transformation_deg,
                                  name='self_transform')

        self.fnn_out3 = FNN(hidden_layer_sizes=[4*self.embedding_size_deg],
                                  inp_size=4*self.embedding_size_deg,
                                  out_size=self.embedding_transformation_deg,
                                  name='self_transform')

        self.Degree_matrix = []
        self.Adjacency_matrix = []
        self.Laplacian_matrix = []
        self.distances_old = {}
        self.edge_index = []
        self.edge_attr = []
        self.edge_attr_start = []
        self.edge_attr_end = []
        self.subgraph_to_graph = []
        self.subgraph2_to_graph = []
        self.center_idx = []
        self.data_list = []
        self.edge_start = []
        self.edge_end = []
        self.x_start_attr = []
        self.x_end_attr = []
        self.x = {}
        self.z_dis = []
        self.z_dis_new = []
        self.x_embedding = []




    def _init_fnns(self):
        if self.single_layer_perceptron:
            print("Using single layer perceptron.")
            with tf.name_scope('self_transform'):
                self.self_transform = SingleLayerFNN(inp_size=self.embedding_size_deg,
                                                     inp_shape=(self.embedding_size_deg, self.embedding_size_deg),
                                                     name='self_transform')
            for i in range(self.hops + 1):
                current_scope = 'shared' + str(i)
                with tf.name_scope(current_scope):
                    self.fnns[i] = SingleLayerFNN(inp_size=self.embedding_size_deg,
                                                  inp_shape=(self.embedding_size_deg, self.embedding_size_deg),
                                                  name=current_scope)
            with tf.name_scope('positional_awareness'):
                self.fnns['pos'] = SingleLayerFNN(inp_size=self.embedding_size_deg,
                                                  inp_shape=(self.embedding_size_deg, self.embedding_size_deg),
                                                  name='positional_awareness')
        else:
            with tf.name_scope('self_transform'):
                self.self_transform = FNN(hidden_layer_sizes=[self.embedding_size_deg],
                                          inp_size=self.embedding_size_deg,
                                          out_size=self.embedding_transformation_deg,
                                          name='self_transform')
            for i in range(self.hops + 1):
                current_scope = 'shared' + str(i)
                with tf.name_scope(current_scope):
                    self.fnns[i] = FNN(inp_size=self.embedding_size_deg,
                                       hidden_layer_sizes=[self.embedding_size_deg * (i + 1)],
                                       out_size=self.embedding_transformation_deg,
                                       name=current_scope)

            with tf.name_scope('positional_awareness'):
                self.fnns['pos'] = FNN(hidden_layer_sizes=[self.embedding_size_deg],
                                       inp_size=self.embedding_size_deg,
                                       out_size=self.embedding_transformation_deg,
                                       name='positional_awareness')

        self.policy_dense = tf.keras.layers.Dense(self.embedding_size_deg)
        self.policy_softmax = tf.keras.layers.Softmax()

        self.conv1 = tf.keras.layers.Conv1D(self.embedding_size_deg, kernel_size=5, strides=4, padding='same')

        self.intra = Intra_Patch_Attention(2*self.embedding_size_deg, False)
        # self.cross_attention = CrossAttention(self.n * self.embedding_size_deg, num_heads=self.num_heads)
        # self.learn = LinearAttention(self.n * self.embedding_size_deg, num_heads=self.num_heads)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

        self.dense1 = tf.keras.layers.Dense(self.embedding_size_deg, activation='relu', use_bias=True)
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        # self.dense2 = tf.keras.layers.Dense(self.embedding_size_deg, use_bias=True)
        # self.dropout2 = tf.keras.layers.Dropout(0.1)



    def build(self, G, E, bs=1):

        #---------------------------------------------------------------------------------------------------------
        self.laplace(G)
        # self.Adjacency_matrix = tf.cast(self.Adjacency_matrix, tf.float32)
        #print("=====================================================461")
        #print(self.Adjacency_matrix)
        #num_row, num_cols = self.Adjacency_matrix.shape
        # ---------------------------------------------------------------------------------------------------------


        """
        1. Build an FNN with the E placeholder
        """
        layers = 1
        self.self_transform = E


        for layer in range(layers):
        ###GNN
            self.self_transform = self.fnn.build(self.self_transform)

        ###sph
            # n_hop_adj = self.n_hop_adj(self.Adjacency_matrix)
            # sph = self.process_hop(n_hop_adj, gamma=2.0, hop=3.0)
            # sph = tf.squeeze(sph, axis=1)

            norm = RMSNorm(self.embedding_size_deg)
            self.self_transform = norm(self.self_transform)

            self.mamba = MambaBlock(int(np.size(self.Adjacency_matrix[0])), self.embedding_size_deg, 1)
            self.self_transform = self.mamba.build(self.self_transform, self.Adjacency_matrix)

        ###gcntf
            num_nodes = np.size(self.Adjacency_matrix[0])
            self.Adjacency_matrixs = self.get_n_hop_adj(self.Adjacency_matrix, self.num_hop, weighted=None)

            self.self_transforms = []
            self.intras = []
            for i in range(self.Adjacency_matrixs.shape[0]):
                layer = self.Adjacency_matrixs[i]
                a_cross = self.gcn_cross.build(self.self_transform, tf.cast(layer, tf.float32))
                a_self = self.gcn_self.build(self.self_transform, tf.cast(layer, tf.float32))
                # a = self.gat.build(self.self_transform, layer)

                # norm = RMSNorm(self.n_self * self.embedding_size_deg)
                # self.mamba = MambaBlock(num_nodes, self.n_self * self.embedding_size_deg, 1)
                # a_mamba = norm(self.self_transform)
                # a_self = self.mamba.build(a_mamba, layer)


                # if i == 0:
                #     self.self_transform = self.gcn.build(self.self_transform, tf.cast(layer, tf.float32))
                # else:
                #     self.self_transform = self.gcn1.build(self.self_transform, tf.cast(layer, tf.float32))
                #
                #
                # self.tr = TransformerEncoderLayer(np.size(self.Adjacency_matrix[0]), nhead=1, dim_feedforward=512,
                #                                        dropout=0.3,
                #                                        activation="relu", batch_norm=True, pre_norm=False,
                #                                        gnn_type="gcn", se="gnn", k_hop=2,
                #                                        embedding_size_deg=2*self.embedding_size_deg)
                #
                # a = self.tr.build(a, layer, self.edge_index,
                #                                          subgraph_node_index=None, subgraph_edge_index=None,
                #                                          subgraph_edge_attr=None,
                #                                          subgraph_indicator_index=None,
                #                                          edge_attr=None, degree=None, ptr=None,
                #                                          return_attn=False
                #                                          )
                # learn = LinearAttention(2 * self.embedding_size_deg, num_heads=2)
                # a = learn(tf.expand_dims(a, axis=0))
                # a = tf.squeeze(a, axis=0)

                ###############################
                # if i == 0:
                #     b = tf.expand_dims(a, axis=0)
                # a1 = tf.expand_dims(a, axis=0)
                # output, attention = self.intra(a1, a1, b)
                # b = (output + a1)/2

                ###############################  1
                if self.type == 'CROSS' or self.type == 'ADD':
                    # print("CROSS")
                    if i < self.cross_hop:
                        mean_values = tf.expand_dims(tf.reduce_mean(a_cross, axis=0), axis=0)
                        b = tf.expand_dims(tf.concat([mean_values, a_cross], axis=0), axis=0)

                        if i == 0: a2 = b
                        # b = (a2 + b)/2  ######

                        a2 = tf.squeeze(self.cross_attention(b, a2), axis=0)
                        a2 = (mean_values + a2) / 2
                        a2 = tf.concat([a2, a_cross], axis=0)

                        # a2 = self.fnn_out2.build(a2)  #######

                        a2 = tf.expand_dims(a2, axis=0)

                ####################################################
                self.self_transforms.append(a_self)
                # self.intras.append(tf.squeeze(output, axis=0))

            self.self_transforms = self.self_transforms[:self.self_hop]
            self.self_transforms = tf.stack(self.self_transforms, axis=0)
            # self.intras = tf.stack(self.intras, axis=0)

            ######################################
            # b = tf.squeeze(b, axis=0)
            # self.self_transform = self.fnn_out.build(b)

            ######################################  1
            if self.type == 'CROSS':
                a2 = tf.squeeze(a2, axis=0)
                self.self_transform = self.fnn_out_cross.build(a2)

        ####transformer new
            if self.type == 'SELF':
                self.self_transforms = self.learn(self.self_transforms)
                self.self_transform = tf.reduce_sum(self.self_transforms, axis=0)
                # self.self_transform = self.self_transform/3
                self.self_transform = self.fnn_out_self.build(self.self_transform)

            #######################################
            if self.type == 'ADD':
                a2 = tf.squeeze(a2, axis=0)
                a2 = self.fnn_out_cross.build(a2)

                self.self_transforms = self.learn(self.self_transforms)
                self.self_transform = tf.reduce_sum(self.self_transforms, axis=0)

                self.self_transform = self.self_transform/self.self_hop

                self.self_transform = self.fnn_out_self.build(self.self_transform)

                mean_values = tf.expand_dims(tf.reduce_mean(self.self_transform, axis=0), axis=0)
                self.self_transform = tf.concat([mean_values, self.self_transform], axis=0)

                self.self_transform = (self.self_transform + a2)/2




            # self.self_transform = self.fnn_out.build(self.self_transform)

            # self.self_transform = tf.concat([self.self_transform, a2], axis=1)
            # self.self_transform = self.fnn_out3.build(self.self_transform)

            # self.self_transform = self.fnn_out.build(tf.concat([self.self_transform, a2], axis=1))

            # self.self_transform1 = self.self_transform + a2
            # self.self_transform2 = self.dense1(self.self_transform1)
            # self.self_transform2 = self.dropout1(self.self_transform2)
            # # self.self_transform2 = self.dense2(self.self_transform2)
            # # self.self_transform2 = self.dropout2(self.self_transform2)
            # self.self_transform = self.self_transform1 + self.self_transform2

            # self.self_transform = self.gcn.build(self.self_transform, tf.cast(self.Adjacency_matrix, tf.float32))  # nmt
            # self.self_transform = self.gat.build(self.self_transform, self.Adjacency_matrix, sph)  # ptb  cifar10

        ###TRANSFORMER
            # self.transformer = TransformerEncoderLayer(np.size(self.Adjacency_matrix[0]), nhead=1, dim_feedforward=512,
            #                                        dropout=0.3,
            #                                        activation="relu", batch_norm=True, pre_norm=False,
            #                                        gnn_type="gcn", se="gnn", k_hop=2,
            #                                        embedding_size_deg=self.embedding_size_deg)
            #
            # self.self_transform = self.transformer.build(self.self_transform, self.Adjacency_matrix, self.edge_index,
            #                                          subgraph_node_index=None, subgraph_edge_index=None,
            #                                          subgraph_edge_attr=None,
            #                                          subgraph_indicator_index=None,
            #                                          edge_attr=None, degree=None, ptr=None,
            #                                          return_attn=False
            #                                          )


          ###TreeScan
            # treescan = TreeScan(self.embedding_size_deg,
            #                 self.embedding_size_deg,
            #                 self.embedding_size_deg,
            #                 self.dropout_rate,
            #                 num_nodes=len(self.Adjacency_matrix),
            #                 arr=self.Adjacency_matrix,
            #                 depths=[3, 4, 3],
            #                 type=None  #'mamba','linear', 'gat', None
            #                 )
            # self.self_transform = treescan.build(self.self_transform,
            #                                      edge_index=self.edge_index,
            #                                      edge_attr=tf.cast(self.Adjacency_matrix, tf.float32))

            # self.self_transform2 = tf.transpose(self.self_transform2, perm=(1, 0))
            # self.conv2 = tf.keras.layers.Dense((self.self_transform1.shape)[0])
            # self.self_transform2 = self.conv2(self.self_transform2)
            # self.self_transform2 = tf.transpose(self.self_transform2, perm=(1, 0))


            # self.self_transform = 0.6*self.self_transform1 + 0.4*self.self_transform2



        ###COGNN

            # I = np.eye(len(self.Adjacency_matrix))
            # new_Adjacency_matrix = I+self.Adjacency_matrix
            # one = list(zip(new_Adjacency_matrix.nonzero()[0], self.Adjacency_matrix.nonzero()[1]))
            # for i in enumerate(one):
            #     self.edge_start.append(i[1][0])
            #     self.edge_end.append(i[1][1])
            #
            # self.edge_index.append(self.edge_start)
            # self.edge_index.append(self.edge_end)  # edge_index  #self.dropout_rate
            #
            # cognn = CoGNN(self.embedding_size_deg,
            #               self.embedding_size_deg,
            #               self.embedding_size_deg,
            #               self.dropout_rate,
            #               num_layers=3,
            #               num_nodes=len(self.Adjacency_matrix),
            #               arr=self.Adjacency_matrix,
            #               skip=True,
            #               type='transformer')
            # self.self_transform = cognn.build(self.self_transform, self.edge_index, edge_attr=tf.cast(self.Adjacency_matrix, tf.float32))

        ###CAGNN   nmt 7 2  cifar 14 1  ptb 7 3
            # self.cagnn = CAGNN(tf.cast(self.Adjacency_matrix, tf.float32),self.Adjacency_matrix, self.embedding_size_deg,
            #                    self.embedding_size_deg,
            #                    self.embedding_transformation_deg,
            #                    num_layers=1,
            #                    dropout_rate=self.dropout_rate)
            # self.self_transform2 = self.cagnn.build(self.self_transform)
            #
            # self.self_transform = tf.concat([self.self_transform1, self.self_transform2], axis=-1)
            # self.self_transform = self.policy_dense(self.self_transform)

            # self.self_transform = (self.self_transform1+self.self_transform2)/2


        # print("self_transform", self.self_transform)
        # x = tf.expand_dims(self.self_transform, axis=0)
        #
        # gru_cell = tf.keras.layers.GRU(int(x.shape[-1]), return_sequences=True, return_state=True)
        # zero_state = gru_cell.get_initial_state(x)
        # output, h = gru_cell(x, zero_state)
        #
        # mu = tf.squeeze(h, axis=0)
        #
        # logvar = tf.reduce_sum(output, axis=1)
        #
        # std = tf.exp(0.5 * logvar)
        # eps = tf.random.normal(tf.shape(std))
        # z = eps * std + mu
        #
        # print("z", z)
        # x = tf.squeeze(x, axis=0)
        # print("x", x)
        # x = 0.9 * x + 0.1 * z
        # print("x", x)
        #
        # self.self_transform = self.policy_dense(x)
        # # self.self_transform = self.policy_softmax(policy_logits)
        #
        # print("self_transform", self.self_transform)


            ###GCN
            # self.self_transform = self.gcn.build(self.self_transform, tf.cast(self.Adjacency_matrix, tf.float32))

            ###GAT
            #self.self_transform = self.gat.build(self.self_transform, self.Adjacency_matrix)

            # ###ST_GNN
            # self.st_gnn = ST_GNN_Model(in_channels=self.embedding_size_deg,
            #                            out_channels=self.embedding_size_deg,
            #                            n_nodes=self.Adjacency_matrix.shape[0],
            #                            gru_hs_l1=16,
            #                            gru_hs_l2=16)
            # self.self_transform = self.st_gnn.build(self.self_transform, self.Adjacency_matrix)

            # ###TRANSFORMER
            # I = np.eye(len(self.Adjacency_matrix))
            # new_Adjacency_matrix = I+self.Adjacency_matrix
            # one = list(zip(new_Adjacency_matrix.nonzero()[0], self.Adjacency_matrix.nonzero()[1]))
            # for i in enumerate(one):
            #     self.edge_start.append(i[1][0])
            #     self.edge_end.append(i[1][1])
            #
            # self.edge_index.append(self.edge_start)
            # self.edge_index.append(self.edge_end)  # edge_index
            #
            # self.transformer = TransformerEncoderLayer(np.size(self.Adjacency_matrix[0]), nhead=1, dim_feedforward=512, dropout=0.3,
            #     activation="relu", batch_norm=True, pre_norm=False,
            #     gnn_type="gcn", se="gnn", k_hop=2,embedding_size_deg=self.embedding_size_deg)
            #
            # self.self_transform = self.transformer.build(self.self_transform, self.Adjacency_matrix, self.edge_index,
            # subgraph_node_index=None, subgraph_edge_index=None,
            # subgraph_edge_attr=None,
            # subgraph_indicator_index=None,
            # edge_attr=None, degree=None, ptr=None,
            # return_attn=False,
            # )

            # self.p = self.gcn.build(self.p, tf.cast(self.Adjacency_matrix, tf.float32))
            # #self.p = self.gat.build(self.self_transform, self.Adjacency_matrix)
            #
            # norm = RMSNorm(self.embedding_size_deg)
            # self.self_transform = norm(self.self_transform)
            # self.p = norm(self.p)
            #
            # self.self_transform = tf.nn.dropout(self.self_transform, 0.1)
            # self.p = tf.nn.dropout(self.p, 0.1)
            #
            # #################    NMT    CIFAR10
            # self.self_transform = 0.2 * self.p + 0.8 * self.self_transform
            #
            # #################    PTB
            # self.self_transform = tf.concat([self.p, self.self_transform], axis=1)
            #
            # self.self_transform = tf.nn.dropout(self.self_transform, 0.1)
            # self.w1 = tf.keras.layers.Dense(self.embedding_size_deg)
            # self.self_transform = self.w1(self.self_transform)
            #
            # self.self_transform = norm(self.self_transform)




            ###Mamba

            # from subgraph import data
            # x1, x2, x3 = data(G, self.self_transform, self.embedding_size_deg)
            #
            # norm = RMSNorm(self.embedding_size_deg)
            # self.mamba = MambaBlock(int(np.size(self.Adjacency_matrix[0])), self.embedding_size_deg, 1)
            #
            # self.x1 = norm(x1)
            # self.x1 = self.mamba.build(self.x1, self.Adjacency_matrix)
            #
            # self.x2 = norm(x2)
            # self.x2 = self.mamba.build(self.x2, self.Adjacency_matrix)
            #
            # self.x3 = norm(x3)
            # self.x3 = self.mamba.build(self.x3, self.Adjacency_matrix)
            #
            # self.x1 = norm(self.x1)
            # self.x3 = norm(self.x3)
            # self.self_transform = norm(self.self_transform)
            #
            # p = tf.concat([self.x1, self.x2, self.x3], axis=1)
            # p = tf.nn.dropout(p, 0.1)
            # self.w1 = tf.keras.layers.Dense(self.embedding_size_deg)
            # self.self_transform = self.w1(p)
            # self.x2 = norm(self.x2)



            # self.self_transform = tf.concat([self.self_transform, p], axis=1)
            # self.self_transform = tf.nn.dropout(self.self_transform, 0.1)
            # self.w2 = tf.keras.layers.Dense(self.embedding_size_deg)
            # self.self_transform = self.w2(self.self_transform)

           #######################################################################


            # norm = RMSNorm(self.embedding_size_deg)
            # self.self_transform = norm(self.self_transform)

            # self.mamba = MambaBlock(int(np.size(self.Adjacency_matrix[0])), self.embedding_size_deg, 1)
            # self.self_transform = self.mamba.build(self.self_transform, self.Adjacency_matrix)


            ###DGNN
            # self.dgnn = DGNN(int(np.size(self.Adjacency_matrix[0])), 1, self.embedding_size_deg,1)
            # self.self_transform = self.dgnn.build(self.self_transform, self.Adjacency_matrix)


            ###DGNN_SUB
            # self.dgnn = DGNN(int(np.size(self.Adjacency_matrix[0])), 1, self.embedding_size_deg, 2, G)
            # self.self_transform = self.dgnn.build(self.self_transform, self.Adjacency_matrix)

            ###ST_GNN
            # self.st_gnn = ST_GNN_Model(in_channels=self.embedding_size_deg,
            #                            out_channels=self.embedding_size_deg,
            #                            n_nodes=self.Adjacency_matrix.shape[0],
            #                            gru_hs_l1=16,
            #                            gru_hs_l2=16)
            # self.self_transform = self.st_gnn.build(self.self_transform, self.Adjacency_matrix)




        # focus = Focus(self.self_transform.shape,self.embedding_size_deg)
        # self.self_transform = focus.build(self.self_transform)
        #
        #
        # self.Adjacencyatrix = self.Adjacenacy_matrix+2*np.eye(len(self.Adjacency_matrix))
        # self.self_transform = self.Adjacency_matrix @ self.self_transform


        #I2GNN--------------------------------------------------------------------------------------------------------

        # I = np.eye(len(self.Adjacency_matrix))
        # new_Adjacency_matrix = I+self.Adjacency_matrix
        # #
        # one = list(zip(new_Adjacency_matrix.nonzero()[0], self.Adjacency_matrix.nonzero()[1]))
        # for i in enumerate(one):
        #     self.edge_start.append(i[1][0])
        #     self.edge_end.append(i[1][1])
        #
        #     diff = new_Adjacency_matrix[i[1][1]] - new_Adjacency_matrix[i[1][0]]
        #     dist = np.linalg.norm(diff, ord=2, axis=-1, keepdims=True)
        #     self.z_dis.append(diff)
        # pca_dist = PCA(n_components=self.embedding_size_deg+2)
        # self.z_dis = pca_dist.fit_transform(self.z_dis)
        #
        # self.edge_index.append(self.edge_start)
        # self.edge_index.append(self.edge_end)  # edge_index
        # sum =0
        #
        # for n in G.nodes():
        #     embedding = tf.expand_dims(self.self_transform[G.get_idx(n), :], axis=0)
        #     self.x[n] = embedding
        #     sum+=1
        #
        # #########################################
        # new_x, new_edge_index, new_z, new_node_to_subgraph2, node_to_original_node, center_index, new_subgraph2_rd, self.new_z2  = self.create_2subgraph(
        #     x=self.self_transform, edge_index=self.edge_index, num_nodes=sum, h=1, sample_ratio=1.0,
        #     max_nodes_per_hop=None,
        #     node_label='spd', use_rd=True, subgraph_pretransform=None, center_idx=True, self_loop=False,
        #     subgraph2_x_select="NONE",input_size=self.embedding_size_deg)
        #
        #
        # self.self_transform = self.i2TF.build(G, new_x, new_edge_index, new_z, new_node_to_subgraph2, node_to_original_node, center_index, new_subgraph2_rd, self.new_z2)
        # #########################################

        # self.self_transform = self.i2gnn.build(G, self.x, self.edge_index, self.z_dis)

        # I2GNN--------------------------------------------------------------------------------------------------------


        """
        2. Generate samples of each node's neighbourhood. Based on the `sample_ratio` class parameter
        """
        #self._generate_samples(G)

        """
        3. Given that we have the of each node and its neighbourhood sample, we generate embeddings for nodes located 
        n hops away.
        """

        # print("开始调用子图")
        # s = data(G, self.self_transform, self.embedding_size_deg, self.aggregation, self.embs)
        # print("结束调用子图")
        #sum = 0

       # for n in G.nodes():
        #     sum = sum+1
        #     ##################
        #    embedding = tf.expand_dims(self.self_transform[G.get_idx(n), :], axis=0)
        #     index = G.get_idx(n)
        #     # embedding = tf.concat((embedding,tf.reshape(s[index], shape=[-1, self.embedding_transformation_deg])), axis=-1)
        #     # embedding = self.base_gcn_pos.build(embedding)
        #     embedding = tf.nn.dropout(embedding, keep_prob=self.dropout_rate)
        #     embedding = tf.nn.l2_normalize(embedding)
        #
        #    self.samples[n] = embedding
            #########################


        # for t in range(2):
        #     for n in G.nodes():
        #         sum = sum + 1
        #         ##################
        #         if t == 0:
        #             embedding = tf.expand_dims(self.self_transform[G.get_idx(n), :], axis=0)
        #             index = G.get_idx(n)
        #
        #
        #             concatenated_with_current = tf.reshape(tf.concat((tf.reshape(s[index], shape=[-1, self.embedding_transformation_deg]), embedding), axis=1),
        #                                                    shape=[-1,self.embedding_size_deg])
        #             embedding = self.fnns[t].build(concatenated_with_current)
        #
        #             embedding = tf.nn.dropout(embedding, keep_prob=self.dropout_rate)
        #             self.samples[n] = embedding
        #             #########################
        #         else:
        #             embedding = tf.expand_dims(self.self_transform[G.get_idx(n), :], axis=0)
        #
        #             concatenated_with_current = tf.concat((self.samples[n], embedding),axis=0)
        #
        #             embedding = self.fnns[t].build(concatenated_with_current)
        #             if t ==1:
        #                 embedding = tf.nn.l2_normalize(embedding)
        #             embedding = tf.nn.dropout(embedding, keep_prob=self.dropout_rate)
        #             self.samples[n] = embedding



        # for i in range(0, self.hops + 1):
        #     for n in G.nodes():
        #         index = G.get_idx(n)
        #
        #         if i == 0:
        #             self.samples[n] = tf.expand_dims(self.self_transform[G.get_idx(n), :], axis=0)
        #
        #         embedding = tf.concat((self.samples[n], tf.reshape(s[index], shape=[-1, self.embedding_transformation_deg])),axis=-1)
        #         embedding = self.base_gcn_pos.build(embedding)
        #
        #         if i == self.hops :
        #             embedding = tf.nn.l2_normalize(embedding)
        #             embedding = tf.nn.relu(embedding)
        #         embedding = tf.nn.dropout(embedding, keep_prob=self.dropout_rate)
        #         self.samples[n] = embedding



        # print("PCA")
        # self.distances_numpy = np.zeros([len(G.nodes()), len(G.nodes())])
        # self.distances_numpy_max = np.zeros([len(G.nodes()), len(G.nodes())])
        # self.nodes = np.zeros([len(G.nodes()), self.embedding_size_deg])
        # self.distances = dict(nx.floyd_warshall(G.G))


        # for n in G.G.nodes():
        #     i = G.get_idx(n)
        #     for ns in self.distances[n].keys():
        #         j = G.get_idx(ns)
        #
        #         if self.distances[n][ns] == "inf":
        #             self.distances[n][ns] = -1.5
        #             self.distances_numpy[i][j] = -1.5
        #
        #
        #         else:
        #             self.distances_numpy[i][j] = self.distances[n][ns]
        #
        # self.distances_numpy_max = self.distances_numpy[np.isinf(self.distances_numpy)] = -1.5
        # max_distances = np.nanmax(self.distances_numpy_max, axis=0, keepdims=True)
        # self.distances_numpy /= max_distances
        # self.distances_numpy = np.cos(self.distances_numpy * np.pi)
        # self.distances_numpy[np.isinf(self.distances_numpy)] = -1.5
        #
        # pca = PCA(n_components=self.embedding_size_deg)
        # ####################################
        # pca_matrix = pca.fit_transform(self.distances_numpy)
        #
        #
        # tmp_matrix = tf.convert_to_tensor(pca_matrix)
        #
        # tmp_matrix = tf.cast(tmp_matrix, tf.float32)
        # tmp_matrix = self.base_gcn.build(tmp_matrix)
        # tmp_matrix = tf.nn.dropout(tmp_matrix, keep_prob=self.dropout_rate)

        # print("laplace")
        # self.Degree_matrix = np.zeros((len(G.nodes()), len(G.nodes())))
        # self.Adjacency_matrix = np.zeros((len(G.nodes()), len(G.nodes())))
        # for n in G.nodes():
        #     for ns in G.neighbors(n):
        #         self.Degree_matrix[G.get_idx(n)][G.get_idx(n)] = self.Degree_matrix[G.get_idx(n)][G.get_idx(n)]+1
        #         self.Adjacency_matrix[G.get_idx(n)][G.get_idx(ns)] = 1
        # #self.element(self.degree_matrix)
        #
        # self.element(self.Adjacency_matrix)


        #self.laplace(G)
        #self.element(self.Laplacian_matrix)

        ##########################################
        #FNN(L)

        # self.base_gcn_pos_ii = SingleLayerFNN(inp_size=self.embedding_size_deg,
        #                                       inp_shape=(len(G.nodes()),self.embedding_size_deg),
        #                                       name='Laplacian_matrix')
        # self.Laplacian_matrix = self.Laplacian_matrix.astype(np.float32)
        # self.Laplacian_matrix = self.base_gcn_pos_ii.build(self.Laplacian_matrix)
        # self.tmp_matrix = tf.convert_to_tensor(self.Laplacian_matrix)
        # self.tmp_matrix = tf.cast(self.tmp_matrix, tf.float32)
        # self.tmp_matrix = tf.nn.dropout(self.tmp_matrix, keep_prob=self.dropout_rate)

        # self.element(self.Laplacian_matrix)
        ##########################################
        #PCA(L)
        #
        # pca = PCA(n_components=self.embedding_size_deg)
        # #pca = PCA(n_components=3)
        # pca_matrix = pca.fit_transform(self.Laplacian_matrix)
        # self.tmp_matrix = tf.convert_to_tensor(pca_matrix)
        # self.tmp_matrix = tf.cast(self.tmp_matrix, tf.float32)
        # self.tmp_matrix = tf.nn.dropout(self.tmp_matrix, keep_prob=self.dropout_rate)
        # print("PCA:self.tmp_matrix"+str(self.tmp_matrix))

        #self.element(pca_matrix)
        ##########################################
        #SVD(L)

        # svd = TruncatedSVD(n_components=self.embedding_size_deg)
        # svd_matrix = svd.fit_transform(self.Laplacian_matrix)
        # self.tmp_matrix = tf.convert_to_tensor(svd_matrix)
        # self.tmp_matrix = tf.cast(self.tmp_matrix, tf.float32)
        # self.tmp_matrix = tf.nn.dropout(self.tmp_matrix, keep_prob=self.dropout_rate)
        # print("SVD:self.tmp_matrix" + str(self.tmp_matrix))

        #self.element(svd_matrix)
        ##########################################
        #ICA(L)

        # ica = FastICA(n_components=self.embedding_size_deg)
        # ica_matrix = ica.fit_transform(self.Laplacian_matrix)
        # self.tmp_matrix = tf.convert_to_tensor(ica_matrix)
        # self.tmp_matrix = tf.cast(self.tmp_matrix, tf.float32)
        # self.tmp_matrix = tf.nn.dropout(self.tmp_matrix, keep_prob=self.dropout_rate)
        # print("ICA:self.tmp_matrix" + str(self.tmp_matrix))

        #self.element(svd_matrix)
        ##########################################
        #TSVD(L)

        # U, S, Vt = np.linalg.svd(self.Laplacian_matrix)
        # k = self.embedding_size_deg # 保留前N个最大的奇异值
        # self.tmp_matrix = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :k]
        # self.tmp_matrix = tf.convert_to_tensor(self.tmp_matrix)
        # self.tmp_matrix = tf.cast(self.tmp_matrix, tf.float32)
        # print("TSVD:self.tmp_matrix" + str(self.tmp_matrix))

        ############################################
        #NMF(L)

        # nmf = NMF(n_components=self.embedding_size_deg)
        # nmf_matrix = nmf.fit_transform(self.Laplacian_matrix)
        # self.tmp_matrix = tf.convert_to_tensor(nmf_matrix)
        # self.tmp_matrix = tf.cast(self.tmp_matrix, tf.float32)
        # self.tmp_matrix = tf.nn.dropout(self.tmp_matrix, keep_prob=self.dropout_rate)
        # print("NMF:self.tmp_matrix" + str(self.tmp_matrix))

        ############################################
        # tsne = TSNE(n_components=3)
        # tsne_matrix= tsne.fit_transform(self.Laplacian_matrix)
        # self.tmp_matrix = tf.convert_to_tensor(tsne_matrix)
        # self.tmp_matrix = tf.cast(self.tmp_matrix, tf.float32)
        # self.tmp_matrix = tf.nn.dropout(self.tmp_matrix, keep_prob=self.dropout_rate)
        # print("TSNE:self.tmp_matrix"+str(self.tmp_matrix))
        ###############################################
        # kpca = KernelPCA(kernel='rbf',gamma=10,n_components=3)
        # kpca_matrix= kpca.fit_transform(self.Laplacian_matrix)
        # self.tmp_matrix = tf.convert_to_tensor(kpca_matrix)
        # self.tmp_matrix = tf.cast(self.tmp_matrix, tf.float32)
        # self.tmp_matrix = tf.nn.dropout(self.tmp_matrix, keep_prob=self.dropout_rate)
        # print("TSNE:self.tmp_matrix"+str(self.tmp_matrix))


        """
        4. Return either the concatenated node embeddings for all nodes for the given number of hops,
        or the P-GNN position aware embeddings based on anchor sets
        """

        if self.position_aware:


            # out = tf.concat([self.samples[n] for n in G.nodes()], axis=0)
            # # self.Adjacency_matrix = tf.cast(self.Adjacency_matrix, tf.float32)
            # # out = self.gcn.build(out, self.Adjacency_matrix)
            #
            # for n in G.nodes():
            #     i = G.get_idx(n)
            #     self.samples[n] = tf.reshape(out[i], [-1, self.embedding_size_deg])


            """
           4.1.1. Pre-calculate distances between all node pairs
            """
            if self.position == 1:
                self._precalculate_distances(G.G, self.pgnn_neigh_cutoff)
                print("PGNN")
                """
                4.1.2. Build the anchor sets based on the Bourgain theorem used in P-GNN
                """
                self._build_anchor_sets(G)

                """
                4.1.3. Generate all embeddings for nodes based on feature info of the node and feature info of the nodes in
                all anchor sets. Anchor set aggregations can be obtained using max or mean aggregation
                """
                positional_info_generator = self._aggregate_positional_info(G.nodes(), self.pgnn_aggregation)

                positions = [pos for pos in positional_info_generator]

                out = tf.reshape(positions, shape=[-1, self.embedding_transformation_deg])
                print("Returning P-GNN values with shape", out.shape, datetime.datetime.now())

                return out
            if self.position == 2:
                self._precalcute_distances_PADEL(G.G, self.pgnn_neigh_cutoff)
                """
                4.1.2. Build the anchor sets based on the Bourgain theorem used in P-GNN
                """
                self._build_anchor_sets(G)

                """
                4.1.3. Generate all embeddings for nodes based on feature info of the node and feature info of the nodes in
                all anchor sets. Anchor set aggregations can be obtained using max or mean aggregation
                """
                positional_info_generator = self._aggregate_positional_info(G.nodes(), self.pgnn_aggregation)

                positions = [pos for pos in positional_info_generator]

                out = tf.reshape(positions, shape=[-1, self.embedding_transformation_deg])
                print("Returning P-GNN values with shape", out.shape, datetime.datetime.now())

                return out
            if self.position == 3:
                self._precalcute_distances_PADEL_all(G.G, sum)
                print("PADEL_ALL")
                """
                4.1.2. Build the anchor sets based on the Bourgain theorem used in P-GNN
                """
                self._build_anchor_sets(G)

                """
                4.1.3. Generate all embeddings for nodes based on feature info of the node and feature info of the nodes in
                all anchor sets. Anchor set aggregations can be obtained using max or mean aggregation
                """
                positional_info_generator = self._aggregate_positional_info(G.nodes(), self.pgnn_aggregation)

                positions = [pos for pos in positional_info_generator]

                out = tf.reshape(positions, shape=[-1, self.embedding_transformation_deg])
                print("Returning P-GNN values with shape", out.shape, datetime.datetime.now())

                return out

            if self.position == 4:
                print("LAPLACE")
                positional_info_generator = self._aggregate_positional_info(G.nodes(), self.pgnn_aggregation)

                positions = [pos for pos in positional_info_generator]

                out = tf.reshape(positions, shape=[-1, self.embedding_transformation_deg])



        else:
            print('Returning GraphSAGE concatenated values.', datetime.datetime.now())

            # out = tf.concat([self.samples[nfor n in G.nodes()], axis=0)
            # return out

            out = self.self_transform

            # out = tf.concat((out, self.tmp_matrix), axis=-1)   #self.Laplacian_matrix   self.tmp_matrix
            #
            # # focus = Focus(out.shape, self.embedding_size_deg)
            # # out = focus.build(out)
            #
            # out = tf.keras.layers.Dense(self.embedding_size_deg)(out)

            #out = self.base_gcn_pos.build(out)
            out = tf.nn.dropout(out, keep_prob=self.dropout_rate)
            out = tf.nn.l2_normalize(out)
            return out

            ##########################################
            # out = tf.concat([self.samples[n] for n in G.nodes()], axis=0)
            # out = tf.concat((out, self.Laplacian_matrix), axis=-1)
            # out = self.base_gcn_pos.build(out)
            #
            # out = tf.nn.l2_normalize(out)
            # out = tf.nn.dropout(out, keep_pro=self.dropout_rate)
            # out = tf.nn.l2_normalize(out)
            # return out
            ##########################################

            ##########################################
            # out = tf.concat([self.samples[n] for n in G.nodes()], axis=0)
            #
            # self.tmp_matrix = self.tmp_matrix * out
            #
            # out = tf.concat((out, self.tmp_matrix), axis=-1)
            # out = self.base_gcn_pos.build(out)
            # out = tf.nn.dropout(out, keep_prob=self.dropout_rate)
            # out = tf.nn.l2_normalize(out)
            # return out
            ##########################################

            # out = tf.concat([self.samples[n] for n in G.nodes()], axis=0)
            # self.Adjacency_matrix = tf.cast(self.Adjacency_matrix, tf.float32)
            # out = self.gcn.build(out, self.Adjacency_matrix)
            #
            # out = tf.concat((out, self.tmp_matrix), axis=-1)
            # out = self.base_gcn_pos.build(out)
            # out = tf.nn.dropout(out, keep_prob=self.dropout_rate)
            # out = tf.nn.l2_normalize(out)
            # return out

    def create_2subgraph(self, x, edge_index, num_nodes, h=1, sample_ratio=1.0, max_nodes_per_hop=None,
                     node_label='hop', use_rd=False, subgraph_pretransform=None, center_idx=True, self_loop=False, subgraph2_x_select=None,input_size=None):

        if type(h) == int:
            h = [h]

        new_data_multi_hop = {}
        self.relabels = []
        self.nodes_sum = []
        self.new_edge_index = []
        self.new_edge_index0 = []
        self.new_edge_index1 = []
        self.new_z = []
        self.node_to_original_node = None




        self.new_subgraph2_x = {}
        self.new_subgraph2_x_sum = []
        self.new_subgraph2_rd = []
        self.new_center_idx = []
        self.new_center_idx0 = []
        self.new_center_idx1 = []
        self.new_node_to_subgraph2 = []
        self.new_z1_ = []
        self.new_subgraph2_edge_index = []
        self.new_subgraph2_edge_index0 = []
        self.new_subgraph2_edge_index1 = []
        self.new_z2 = []

        for h_ in h:
            subgraphs = []
            for ind in range(num_nodes):
                nodes_, edge_index_, edge_mask_, z_, relabel = self.k_hop_subgraph(ind, h_, edge_index, True, num_nodes,
                                                                                   node_label=node_label,
                                                                                   max_nodes_per_hop=max_nodes_per_hop)

                for nodes_i in enumerate(nodes_):
                    self.nodes_sum.append(nodes_i[1])

                edge_index_i = 0
                for edge_index_elem in edge_index_:
                    if edge_index_i % 2 == 0:
                        for edge_index0 in edge_index_elem:
                            self.new_edge_index0.append(edge_index0)
                    else:
                        for edge_index1 in edge_index_elem:
                            self.new_edge_index1.append(edge_index1)
                    edge_index_i += 1

                self.add_elem(z_, self.new_z)

                x_ = None
                edge_attr_ = None
                pos_ = None
                for nodes_single in range(len(nodes_)):
                    if x is not None:
                        c = tf.reshape(x[nodes_[nodes_single]], shape=[-1, self.embedding_transformation_deg])
                        if x_ == None:
                            x_ = c
                        else:
                            x_ = tf.concat([x_, c], axis=0)
                    else:
                        x_ = None

                num_edges_ = len(edge_index_[0])

                # node_type edge_attr pos  shuhao 20233028

                num_nodes_ = len(nodes_)
                if center_idx:
                    subgraph2_x, subgraph2_rd, center_idx_, node_to_subgraph2, z1_, edge_index1_, nodes1_ = self.subgraph_to_subgraph2_with_idx(
                        nodes_, num_edges_, x_, z_, edge_index_, num_nodes_, h_, node_label=node_label, use_rd=use_rd,
                        self_loop=self_loop, num_hops=h_)

                else:
                    data_ = self.subgraph_to_subgraph2(h_, node_label=node_label, use_rd=use_rd)

                if len(subgraph2_rd) != len(z1_):
                    subgraph2_rd = np.zeros((len(z1_), 2))

                # self.new_subgraph2_edge_index
                edge_index1_i = 0
                for edge_index1_elem in edge_index1_:
                    if edge_index1_i % 2 == 0:
                        for edge_index1_idx0 in edge_index1_elem:
                            self.new_subgraph2_edge_index0.append(edge_index1_idx0)
                    else:
                        for edge_index1_idx1 in edge_index1_elem:
                            self.new_subgraph2_edge_index1.append(edge_index1_idx1)
                    edge_index1_i += 1

                # self.new_subgraph2_x
                self.new_subgraph2_x[ind] = subgraph2_x

                # self.new_subgraph2_rd
                self.add_elem(subgraph2_rd, self.new_subgraph2_rd)

                # self.new_node_to_subgraph2
                self.add_elem(node_to_subgraph2.T, self.new_node_to_subgraph2)

                # self.new_z1_
                self.add_elem(z1_, self.new_z1_)

                row = np.shape(nodes1_)[0]
                if row == 0:
                    cols = 1
                elif isinstance(nodes1_[0], (list, np.ndarray)):
                    cols = len(nodes1_[0])
                else:
                    cols = 1
                new_laplace = np.zeros((row * cols, row * cols))

                laplace_idx = 0
                for matrix in nodes1_:
                    if isinstance(matrix, int):
                        #new_laplace[laplace_idx][laplace_idx] = self.Laplacian_matrix[matrix][matrix]
                        new_laplace[laplace_idx][laplace_idx] = self.Adjacency_matrix[matrix][matrix]
                        laplace_idx += 1
                    else:
                        for i, matrix_node in enumerate(matrix):
                            new_laplace[laplace_idx][laplace_idx] = self.Adjacency_matrix[matrix_node][matrix_node]
                            laplace_idx += 1
                # print(new_laplace)

                if np.shape(self.new_z2)[0] == 0:
                    self.new_z2 = new_laplace
                else:
                    shape1 = np.array(self.new_z2).shape
                    shape2 = np.array(new_laplace).shape

                    # 计算拼接后矩阵的总形状
                    total_rows = shape1[0] + shape2[0]
                    total_cols = shape1[1] + shape2[1]
                    result = np.zeros((total_rows, total_cols))  # 创建一个全零矩阵，形状为(total_rows, total_cols)
                    result[:shape1[0], :shape1[1]] = self.new_z2  # 将matrix1复制到result的左上角
                    result[shape1[0]:, shape1[1]:] = new_laplace  # 将matrix2复制到result的右下角
                    self.new_z2 = result

                # self.new_center_idx
                i = 0
                for elem in center_idx_:
                    if i % 2 == 0:
                        for idx0 in elem:
                            self.new_center_idx0.append(idx0)
                    else:
                        for idx1 in elem:
                            self.new_center_idx1.append(idx1)
                    i += 1

                deg = np.sum(edge_index_[0] == 0)

                if deg == 0:
                    deg = 1
                self.relabels = self.relabels + list(np.array(relabel)) * deg

        self.new_center_idx.append(self.new_center_idx0)
        self.new_center_idx.append(self.new_center_idx1)

        self.new_subgraph2_edge_index.append(self.new_subgraph2_edge_index0)
        self.new_subgraph2_edge_index.append(self.new_subgraph2_edge_index1)

        self.new_edge_index.append(self.new_edge_index0)
        self.new_edge_index.append(self.new_edge_index1)

        self.node_to_original_node = self.relabels

        for o in enumerate(self.new_subgraph2_x.values()):
            self.new_subgraph2_x_sum.append(o[1])

        return self.new_subgraph2_x_sum, self.new_subgraph2_edge_index, self.new_z1_, self.new_node_to_subgraph2, self.node_to_original_node, self.new_center_idx, self.new_subgraph2_rd, self.new_z2






    def add_elem(self, input, output):
        for ele in enumerate(input):
            output.append(ele[1])


    def subgraph_to_subgraph2_with_idx(self, nodes_, num_edges_, x_, z_, edge_index_, num_nodes_, h_, node_label, use_rd, self_loop,num_hops):

        neighbors = []

        for i in enumerate(edge_index_[0]):
            if i[0] == 0:
                center = i[1]
            if i[1] == center:
                neighbors.append(edge_index_[1][i[0]])

        if self_loop :
            neighbors = np.concatenate([neighbors, [0]])
        num_neighbors = len(neighbors)

        if use_rd:
            rd_to_x = np.zeros([num_nodes_ * num_neighbors, 2])
            rd = self.compute_rd(edge_index_, num_nodes_)
        if num_neighbors == 0:
            x_ = x_
            nodes_ = nodes_
        else:
            x_ = tf.tile(x_, [num_neighbors, 1])
            nodes_ = np.tile(nodes_, [num_neighbors, 1])

        if node_label.startswith('spd'):
            z1_ = np.tile(z_, (num_neighbors, 2))
        else:
            z1_ = np.tile(z_, (num_neighbors, 1))
        edge_index1_ = np.zeros([2, num_edges_ * num_neighbors], dtype=np.int64)
        node_to_subgraph2 = np.zeros([num_nodes_ * num_neighbors], dtype=np.int64)

        if num_neighbors == 0:
            node_to_subgraph2 = np.zeros([num_nodes_], dtype=np.int64)
            subgraph2_to_subgraph = np.zeros([1], dtype=np.int64)
            num_subgraphs2 = 1

            if node_label.startswith('spd'):
                z1_ = np.tile(z_, (1,2))
            elif node_label.startswith('dspd'):
                z1_ = np.tile(z_, (1,4))

            if use_rd and rd is not None:
                rd = np.tile(rd[0, :].reshape([-1, 1]), (1, 2))
            else:
                rd = None

            center_idx = [[0], [0]]
            edge_index1_ = np.zeros([2, 1], dtype=np.int64) -1

        else:
            center_idx = [[], []]
            for i, n in enumerate(neighbors):
                if node_label.startswith('spd'):
                    z1_[i * num_nodes_: (i + 1) * num_nodes_] = np.concatenate([z_, z_ + num_hops + 3], axis=-1)
                else:
                    z1_ = z1_

                if use_rd:
                    rd_to_x[i * num_nodes_: (i + 1) * num_nodes_] = np.concatenate(
                        [rd[0, :].reshape(-1, 1), rd[n, :].reshape(-1, 1)], axis=1)

                edge_index_ = np.array(edge_index_)
                temp = edge_index_ + i * num_nodes_

                edge_index1_ = np.concatenate((temp, edge_index1_), axis=-1)

                node_to_subgraph2[i * num_nodes_: (i + 1) * num_nodes_] = i * np.ones([num_nodes_])

                center_idx[0].append(0 + num_nodes_ * i)
                center_idx[1].append(n.item() + num_nodes_ * i)

        if x_ != None:
            subgraph2_x = x_
        if use_rd:
            subgraph2_rd = rd_to_x


        return subgraph2_x, subgraph2_rd, center_idx, node_to_subgraph2, z1_, edge_index1_, nodes_



    def compute_rd(self, edge_index, num_nodes):
        adj  = np.zeros((num_nodes, num_nodes))
        adj[tuple(edge_index)] = 1
        laplacian = np.diag(np.sum(adj, axis=0)) - adj
        try:
            L_inv = np.linalg.pinv(laplacian)
        except:
            laplacian += 0.01 * np.eye(*laplacian.shape)
            L_inv = np.linalg.pinv(laplacian)

        l_diag = L_inv[np.arange(len(L_inv)), np.arange(len(L_inv))]
        l_i = np.tile(np.expand_dims(l_diag, axis=1), [1, len(L_inv)])
        l_j = np.tile(np.expand_dims(l_diag, axis=0), [len(L_inv), 1])
        rd = np.array(l_i +l_j - L_inv -L_inv.T, dtype=np.float32)
        return rd



    def subgraph_to_subgraph2(self, h_, node_label, use_rd):
        print('subgraph_to_subgraph2')



    def k_hop_subgraph(self,node_idx, num_hops, edge_index, relabel_nodes=False,
                   num_nodes=None, flow='source_to_target', node_label='hop',
                   max_nodes_per_hop=None):
        num_nodes = self.maybe_num_nodes(edge_index, num_nodes)

        assert  flow in ['source_to_target', 'target_to_source']
        if flow == 'target_to_source':
            row, col = edge_index
        else:
            col, row = edge_index

        node_mask = np.full(shape=num_nodes, fill_value=False, dtype=bool)
        edge_mask = np.full(shape=len(row), fill_value=False, dtype=bool)
        subsets = [np.array([node_idx])]
        visited = set(subsets[-1].flatten())
        label = defaultdict(list)
        for node in subsets[-1]:
            label[node].append(1)
        if node_label == 'hop':
            hops = [np.array([0])]

        new_nodes = []

        for h in range(num_hops):
            #node_mask.fill(False)
            node_mask[np.array(subsets[-1], dtype=int)] = True
            #edge_mask = node_mask[row]
            edge_mask = np.take(node_mask, row, out=edge_mask)
            for i, edge in enumerate(edge_mask):
                if edge:
                    new_nodes = np.concatenate((new_nodes, [col[i]]))
            tmp = []
            new_nodes = [int(new_nodes_x) if isinstance(new_nodes_x, float) else new_nodes_x for new_nodes_x in new_nodes]
            for node in new_nodes:
                if node in visited:
                    continue
                tmp.append(node)
                label[node].append(h+2)
            if len(tmp) == 0:
                break
            if max_nodes_per_hop is not None:
                if max_nodes_per_hop < len(tmp):
                    tmp = random.sample(tmp, max_nodes_per_hop)
            new_nodes = set(tmp)
            visited = visited.union(new_nodes)
            #new_nodes = np.array(list(new_nodes))
            new_nodes = list(new_nodes)
            subsets.append(new_nodes)
            if node_label == 'hop':
                hops.append(np.array([h + 1] * len(new_nodes)))
        subset = np.concatenate(subsets)
        subsets_array = np.array(subsets)
        inverse_map = np.arange(subsets_array.shape[0])
        if node_label == 'hop':
            hop = np.concatenate(hops)
        subset = np.delete(subsets, np.where(subset == node_idx))
        subset = np.concatenate(([node_idx], subset))
        flattened_list = subset.flatten().tolist()
        subset = self.use_flatten_list(flattened_list)



        sum_subset = 0
        for i in enumerate(subset):
            sum_subset+=1

        z = None
        if node_label == 'hop':
            hop = hop[hop != 0]
            hop = np.concatenate(([0], hop))
            z = hop.reshape(-1, 1)
            z = np.zeros_like(z)
            z[0, 0] =1.
        elif node_label.startswith('spd') or node_label == 'drnl' or node_label.startswith('dspd'):
            if node_label.startswith('spd'):
                num_spd = int(node_label[3:]) if len(node_label) > 3 else 2
                z = np.zeros([sum_subset, num_spd], dtype=np.long)
            elif node_label.startswith('dspd'):
                num_spd = int(node_label[4:]) if len(node_label) > 4 else 2
                z = np.zeros([sum_subset, num_spd], dtype=np.long)
            elif node_label == 'drnl':
                num_spd = 2
                z = np.zeros([sum_subset, 1], dtype=np.long)

            for i, node in enumerate(subset):
                dists = label[node][:num_spd]
                if node_label == 'spd' or node_label == 'dspd':
                    z[i][:min(num_spd, len(dists))] = np.array(dists[0])
                elif node_label == 'drnl':
                    dist1 = dists[0]
                    dist2 = dists[1] if len(dists) == 2 else 0
                    if dist2 == 0:
                        dist = dist1
                    else:
                        dist = dist1 * (num_hops + 1) + dist2
                    z[i][0] = dist

        node_mask = np.zeros(num_nodes, dtype=bool)
        node_mask[np.array(subset, dtype=int)] = True
        edge_mask = node_mask[row] & node_mask[col]


        sum_edge_index = 0
        for i in enumerate(edge_index[0]):
            sum_edge_index +=1

        new_edge_index_0 = []
        new_edge_index_1 = []
        new_edge_index = []
        for i in range(sum_edge_index):
            if edge_mask[i] == True:
                new_edge_index_0.append(edge_index[0][i])
                new_edge_index_1.append(edge_index[1][i])
        new_edge_index.append(new_edge_index_0)
        new_edge_index.append(new_edge_index_1)
        edge_index = new_edge_index

        sum_edge_index_new = 0
        for i in enumerate(edge_index[0]):
            sum_edge_index_new +=1


        if relabel_nodes:
            node_idx = np.full((num_nodes,), -1)
            node_idx[subset] = np.arange(sum_subset)
            #edge_index = node_idx[edge_index]]
            for i in range(sum_edge_index_new):
                edge_index[0][i] = node_idx[edge_index[0][i]]
                edge_index[1][i] = node_idx[edge_index[1][i]]

        return subset, edge_index, edge_mask, z, subset


    def use_flatten_list(self, input):
        flattened_list = []
        for item in input:
            if isinstance(item, list):
                flattened_list.extend(self.use_flatten_list(item))
            else:
                flattened_list.append(item)
        return flattened_list



    def maybe_num_nodes(self, index, num_nodes=None):
        index_np = np.array(index)
        if num_nodes is None:
            num_nodes = np.max(index_np) + 1
        return num_nodes

    def element(self, matrix):
        row_num = 0
        for row in matrix:
            row_num +=1
            col_num = 0
            for elem in row:
                col_num+=1
                print(float(elem), end=' ')
            print()
        print("row_num:"+str(row_num))
        print("col_num:"+str(col_num))


    def matrix_neg_sqrt(self):
        eigen_values, eigen_vectors = np.linalg.eig(self.Degree_matrix)
        sqrt_eigen_values = np.sqrt(eigen_values)

        D = np.diag(1/sqrt_eigen_values)
        D[np.isinf(D)] = 0

        self.Degree_matrix = eigen_vectors @ D @np.transpose(eigen_vectors)


    def matrix_neg(self):
        eigen_values, eigen_vectors = np.linalg.eig(self.Degree_matrix)
        sqrt_eigen_values = np.sqrt(eigen_values)
        D = np.diag(sqrt_eigen_values)
        D[np.isinf(D)] = 0

        self.Degree_matrix = eigen_vectors @ D @ np.transpose(eigen_vectors)

    def get_n_hop_adj(self, adj_matrix, n, weighted=None):
        """
        根据邻接矩阵获得 n 跳的连接关系，并输出每跳中的连接数
        :param adj_matrix: 邻接矩阵 (numpy array)
        :param n: 跳数
        :return: n 跳的连接关系列表和每跳中的连接数
        """
        adj_matrix = adj_matrix.copy()
        hop_matrices = []
        connection_counts = []
        weighted_hop_matrices = []
        hop = 1

        if weighted == 'sph':
            print("weighted sph")
            for i in range(n):
                if i == 0:
                    hop_matrix = adj_matrix
                else:
                    hop_matrix = tf.matmul(hop_matrix, adj_matrix)
                    hop_matrix = tf.cast(hop_matrix > 0, tf.int32)  # 二值化

                hop_matrices.append(hop_matrix)
                connection_count = tf.reduce_sum(hop_matrix)
                connection_counts.append(connection_count)

                hop_matrix = tf.cast(hop_matrix, dtype=tf.float64)
                if i >= hop:
                    normalized_weights = self.process_hop(hop_matrix, hop=i)
                    weighted_hop_matrix = hop_matrix * normalized_weights
                else:
                    weighted_hop_matrix = hop_matrix
                weighted_hop_matrices.append(weighted_hop_matrix)
            return np.stack(weighted_hop_matrices, axis=0)

        elif weighted == 'weight':
            print("weighted weight")
            for i in range(n):
                if i == 0:
                    hop_matrix = adj_matrix
                else:
                    hop_matrix = np.dot(hop_matrix, adj_matrix)
                    hop_matrix = (hop_matrix > 0).astype(int)  # 二值化

                hop_matrices.append(hop_matrix)
                connection_count = np.sum(hop_matrix)
                connection_counts.append(connection_count)

                max_connection_count = max(connection_counts)
                normalized_weights = 1 - (connection_count / max_connection_count)
                weighted_hop_matrix = hop_matrix * normalized_weights
                weighted_hop_matrices.append(weighted_hop_matrix)
            return np.stack(weighted_hop_matrices, axis=0)

        else:
            print("no weighted")
            for i in range(n):
                if i == 0:
                    hop_matrix = adj_matrix
                else:
                    hop_matrix = np.dot(hop_matrix, adj_matrix)
                    hop_matrix = (hop_matrix > 0).astype(int)  # 二值化

                hop_matrices.append(hop_matrix)
                connection_count = np.sum(hop_matrix)
                connection_counts.append(connection_count)
            return np.stack(hop_matrices, axis=0)


    def n_hop_adj(self, adj_matrix):
        inf = np.inf
        dist_matrix = np.where(adj_matrix > 0, adj_matrix, inf)
        n = dist_matrix.shape[0]
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    dist_matrix[i, j] = min(dist_matrix[i, j], dist_matrix[i, k] + dist_matrix[k, j])

        # 将 dist_matrix 中的 np.inf 转换为 0
        dist_matrix = np.where(np.isinf(dist_matrix), -1.0, dist_matrix)

        return dist_matrix

    def process_hop(self, sph, gamma=2.0, hop=3.0, slope=0.1):
        gamma = tf.cast(gamma,  dtype=sph.dtype)
        # 初始化 LeakyReLU 激活函数，使用自定义的负斜率
        leaky_relu = tf.keras.layers.LeakyReLU(alpha=slope)

        # # 增加 `sph` 张量的维度，以便进行后续操作
        # sph = tf.expand_dims(sph, axis=1)

        # 从 `sph` 中减去 `hop` 值，以计算跳跃的差值
        sph = sph - hop

        # 再次应用 LeakyReLU 激活函数，保持结果的非线性特性
        sph = leaky_relu(sph)

        # 使用 `gamma` 为底，对处理后的 `sph` 进行指数运算，得到最终的跳跃概率
        sp = tf.pow(gamma, sph)

        # 返回最终计算得到的跳跃概率
        return sp

    def laplace(self, G):

        self.Degree_matrix = np.zeros([len(G.nodes()), len(G.nodes())])
        self.Adjacency_matrix = np.zeros([len(G.nodes()), len(G.nodes())])
        for n in G.nodes():
            for ns in G.neighbors(n):
                self.Degree_matrix[G.get_idx(n)][G.get_idx(n)] = self.Degree_matrix[G.get_idx(n)][G.get_idx(n)]+1
                self.Adjacency_matrix[G.get_idx(n)][G.get_idx(ns)] = 1


        #self.Laplacian_matrix = self.Degree_matrix - self.Adjacency_matrix
        # self.laplace_normal(G)
        #self.laplace_randomwalk_normal(G)
        #self.distances_normal(G)


    def distances_normal(self, G):
        self.distances_matrix = np.zeros([len(G.nodes()), len(G.nodes())])
        self.distances = dict(nx.all_pairs_shortest_path_length(G.G, len(G.nodes())))

        for n in G.G.nodes():
            i = G.get_idx(n)
            for ns in self.distances[n].keys():
                j = G.get_idx(ns)
                if self.distances[n][ns] == "inf":
                    self.distances_matrix[i][j] = 0
                else:
                    self.distances_matrix[i][j] = self.distances[n][ns]
        self.Laplacian_matrix = self.distances_matrix




    def laplace_randomwalk_normal(self, G):
        self.matrix_neg()
        self.Laplacian_matrix = self.Degree_matrix @ self.Adjacency_matrix

        #transition_matrix = self.Degree_matrix @ self.Adjacency_matrix
        #I = np.identity(len(G.nodes()))
        #self.Laplacian_matrix = I - np.sqrt(self.Degree_matrix) @ transition_matrix @ np.sqrt(self.Degree_matrix)


    def laplace_normal(self, G):
        I = np.identity(len(G.nodes()))

        self.matrix_neg_sqrt()

        #self.Laplacian_matrix = I - self.Degree_matrix @ self.Adjacency_matrix @ self.Degree_matrix
        self.Laplacian_matrix = self.Degree_matrix @ self.Adjacency_matrix @ self.Degree_matrix
        #self.Laplacian_matrix = self.Adjacency_matrix



    def _aggregate_positional_info(self, nodes, aggregation='max'):
        for i, n in enumerate(nodes):
            if self.memo.get(n) is None:
                self.memo[n] = {}

            positional_aggregation = []
            for anchor_set in self.anchor_sets:
                aggregated = None
                if aggregation == 'max':
                    aggregated = self._max_aggregate_anchor(anchor_set, n)
                elif aggregation == 'mean':
                    # This one has a big performance overhead
                    aggregated = self._mean_aggregate_anchor(anchor_set, n)

                positional_aggregation.append(aggregated)

            positional_aggregation = tf.concat(positional_aggregation, axis=0)
            positional_aggregation = tf.reduce_mean(positional_aggregation, axis=0)
            positional_aggregation = tf.expand_dims(positional_aggregation, axis=0)
            yield self.fnns['pos'].build(positional_aggregation)


    def _mean_aggregate_anchor(self, anchor_set, node):
        node_positions = []
        for anchor in anchor_set:
            if self.memo[node].get(anchor) is not None:
                node_anchor_relation = self.memo[node][anchor]
                node_positions.append(node_anchor_relation)
                continue

            node_embedding = self.samples[node]
            anchor_embedding = self.samples[anchor]

            # positional info between n and anchor node
            if self.distances.get(node) is not None and self.distances[node].get(anchor) is not None:
                positional_info = 1 / (self.distances[node][anchor] + 1)
                feature_info = tf.concat((node_embedding, anchor_embedding), axis=0)
                node_anchor_relation = positional_info * feature_info
            else:
                node_anchor_relation = tf.zeros(shape=[node_embedding.shape[0] + anchor_embedding.shape[0],
                                                       node_embedding.shape[-1]])

            self.memo[node][anchor] = node_anchor_relation
            node_positions.append(node_anchor_relation)
        return tf.reduce_mean(node_positions, axis=0)

    def _max_aggregate_anchor(self, anchor_set, node):
        # find the nodes of the anchor set which can be reached by the current node
        anchor_node_intersections = [(k, self.distances[node][k]) for k in anchor_set
                                     if self.distances[node].get(k) is not None and k != node]

        # get the node with the maximum distance
        max_agg_anchor = max(anchor_node_intersections, key=lambda i: i[1], default=None)

        node_embedding = self.samples[node]
        # if there is no such node, create a zero tensor to keep the dimensions
        if max_agg_anchor is None:
            return tf.zeros(shape=[node_embedding.shape[0] + node_embedding.shape[0],
                                   node_embedding.shape[-1]])

        # get the precalculated embedding of the max node
        anchor_embedding = self.samples[max_agg_anchor[0]]

        # positional_info = 1 / (self.distances[node][max_agg_anchor[0]] + 2)
        #
        # node_anchor_relation = tf.concat((node_embedding * positional_info, anchor_embedding), axis=0)
        #
        # return node_anchor_relation

        if self.distances[node][max_agg_anchor[0]] == -1 :
            positional_info = 0
        else:
            positional_info = 1 / (self.distances[node][max_agg_anchor[0]]+1)

        feature_info = tf.concat((node_embedding, anchor_embedding), axis=0)

        node_anchor_relation = positional_info * feature_info
        return node_anchor_relation


    def _generate_samples(self, G):
        for n in G.nodes():
            if self.samples.get(n) is None:
                self.samples[n] = {}
            self.samples[n] = self._get_sample(G, n)

    def _get_embeddings(self, G, n, level):
        neigh_samples = self.samples[n]['init']
        # if we don't have the initial embeddings of current node and its neighbours

        # TODO maybe move to a separate method?
        """
        3.1.1 If it is the first level, we generate the embedding based only on the fetures of each node.
        We return the embedding of the current node for the current level, as well as the list of its neighbours'
        embeddings for the same level.
        """
        if level == 0:
            self._generate_initial_self_embedding(G, n, self.self_transform)
            for ns in neigh_samples:
                self._generate_initial_self_embedding(G, ns, self.self_transform)

        return self.samples[n][str(level)], [self.samples[neigh][str(level)] for neigh in neigh_samples]

    def _generate_initial_self_embedding(self, G, node, n_transform):
        if self.samples[node].get('0') is None:
            self.samples[node]['0'] = tf.expand_dims(n_transform[G.get_idx(node), :], axis=0)


    def _aggregate_for_node(self, aggregated):
        aggregation = self.aggregation
        if aggregation == 'mean':
            aggregated = tf.reduce_mean(aggregated, axis=0)
        elif aggregation == 'max':
            aggregated = tf.reduce_max(aggregated, axis=0)
        elif aggregation == 'min':
            aggregated = tf.reduce_min(aggregated, axis=0)
        elif aggregation == 'sum':
            aggregated = tf.reduce_sum(aggregated, axis=0)

        return aggregated

    def _get_sample(self, G, node):
        """
        Get a random sample of neighbours based on the ratio
        e.g. if the ratio is 0.5, we will return only half the successors of the current node
        """
        neighbors = [neighbor for neighbor in G.neighbors(node)]
        samples = random.sample(neighbors, int(len(neighbors) * self.sample_ratio))
        return samples

    def _precalculate_distances(self, G, cutoff=6):
        self.distances = dict(nx.all_pairs_shortest_path_length(G, cutoff))


        # self.distances_old = dict(nx.all_pairs_shortest_path_length(G, cutoff))
        # for n in G.nodes():
        #     sum = 0
        #     for name, value in self.distances_old[n].items():
        #         if value == 2:
        #             self.distances[n] = {name:value}
        #             sum = sum+1
        #         if value == 1:
        #             self.distances[n] = {name:value}
        #             sum = sum+1
        #     if sum == 0:
        #         self.distances[n] = {n:0}
        # #print(self.distances)
        # print(len(self.distances))


    def _precalcute_distances_PADEL_all(self, G, cutoff=sum):
        self.distances = dict(nx.all_pairs_shortest_path_length(G, cutoff))
        for n in G.nodes():
            max_distance = max(self.distances[n].values())
            for ns in self.distances[n].keys():
                if self.distances[n][ns] == 0 :
                    self.distances[n][ns] = 0
                else:
                    self.distances[n][ns] = self.distances[n][ns] / max_distance
                    self.distances[n][ns] = np.cos(self.distances[n][ns] * np.pi)

    def _precalcute_distances_PADEL(self, G, cutoff=6):
        print("PADEL")
        print(cutoff)
        self.distances = dict(nx.all_pairs_shortest_path_length(G, cutoff))
        for n in G.nodes():
            max_distance = max(self.distances[n].values())
            ##############################
            # num = 0
            # for ns in self.distances[n].keys():
            #     if self.distances[n][ns] == 0 :
            #         self.distances[n][ns] = 0
            #     elif num%2 == 0:
            #         self.distances[n][ns] = self.distances[n][ns] / max_distance
            #         self.distances[n][ns] = np.sin(self.distances[n][ns] * np.pi)
            #     else:
            #         self.distances[n][ns] = self.distances[n][ns] / max_distance
            #         self.distances[n][ns] = np.cos(self.distances[n][ns] * np.pi)
            #     num = num+1
            ################################
            for ns in self.distances[n].keys():
                if self.distances[n][ns] == 0:
                    self.distances[n][ns] = 0
                else:
                    self.distances[n][ns] = self.distances[n][ns] / max_distance
                    self.distances[n][ns] = np.cos(self.distances[n][ns] * np.pi)
            ################################shuhao



    def PositionEncoding(self, G, cutoff=sum):
        self.distances = dict(nx.floyd_warshall(G))

        for n in G.nodes():

            for ns in self.distances[n].keys():

                if self.distances[n][ns] == 0 :
                    self.distances[n][ns] = 0
                    self.distances_numpy[i][j] = 0
                elif self.distances[n][ns] == "inf" :
                    self.distances[n][ns] =  -1.5
                    self.distances_numpy[i][j] = -1.5
                else:
                    self.distances[n][ns] = self.distances[n][ns] / cutoff
                    self.distances[n][ns] = np.cos(self.distances[n][ns] * np.pi)
                    self.distances_numpy[i][j] = self.distances[n][ns]




    def _build_anchor_sets(self, G, c=0.2):
        n = len(G.nodes())
        m = int(np.log(n))
        copy = int(self.pgnn_c * m)
        for i in range(m):
            anchor_size = int(n / np.exp2(i + self.pgnn_anchor_exponent))

            for j in range(np.maximum(copy, 1)):
                size = random.sample(G.nodes(), anchor_size)
                #for k in size:
                    #print(G.get_idx(k))
                #print("wancheng")

                self.anchor_sets.append(size)
                #print("480")
                #print(size)
        print("Number of anchor sets: ", len(self.anchor_sets),
              ". Biggest set is:" + str(int(n / np.exp2(self.pgnn_anchor_exponent))))

    ''' 
      change this to be generic across different graphs to be placed
    '''


class Messenger(object):

    def __init__(self, d, d1, small_nn=False, dtype=tf.float32):
        # forward pass
        with tf.name_scope('FPA'):
            # self.fpa = Aggregator(d, d1, d1, False, small_nn=small_nn, dtype=dtype)
            self.fpa = Aggregator(d, d1, d1, False, small_nn=small_nn, dtype=dtype)
        with tf.name_scope('BPA'):
            self.bpa = Aggregator(d, d1, d1, False, small_nn=small_nn, dtype=dtype)
            # self.bpa = Aggregator(d, d1, d1, False, small_nn=small_nn, dtype=dtype)
        with tf.name_scope('node_transform'):
            if small_nn:
                self.node_transform = FNN(d, [d], d1, 'fnn', dtype=dtype)
            else:
                self.node_transform = FNN(d, [d, d], d1, 'fnn', dtype=dtype)

    def build(self, G, node_order, E, bs=1):
        try:
            self_trans = self.node_transform.build(E)

            def message_pass(nodes, messages_from, agg):
                node2emb = {}

                for n in nodes:
                    msgs = [node2emb[pred] for pred in messages_from(n)]

                    node2emb[n] = tf.expand_dims(self_trans[G.get_idx(n), :],
                                                 axis=0)

                    if len(msgs) > 0:
                        t = tf.concat(msgs, axis=0)
                        inp = agg.build(t)
                        node2emb[n] += inp

                return tf.concat([node2emb[n] for n in G.nodes()], axis=0)
                # TODO
                # return [node2emb[n] for n in G.nodes()]

            out_fpa = message_pass(node_order, G.predecessors, self.fpa)
            print("Finished forward pass of Messenger")
            out_bpa = message_pass(reversed(node_order), G.neighbors, self.bpa)
            print("Finished backward pass of Messenger")

            out = tf.concat([out_fpa, out_bpa], axis=-1)
            # TODO
            # out = [out_fpa, out_bpa]
            # out = tf.reduce_mean(tf.concat([out_bpa,out_fpa], axis=0), axis=0)
            return out
        except Exception:
            # import my_utils; my_utils.PrintException()
            traceback.print_exc()
            # import pdb
            # pdb.set_trace()


class RadialMessenger(Messenger):

    def __init__(self, k, d, d1, small_nn=False, dtype=tf.float32):
        Messenger.__init__(self, d, d1, small_nn, dtype)
        self.dtype = dtype
        self.k = k

    def build(self, G, f_adj, b_adj, E, bs=1):
        assert np.trace(f_adj) == 0
        assert np.trace(b_adj) == 0

        E = tf.cast(E, dtype=self.dtype)

        E = tf.reshape(E, [-1, tf.shape(E)[-1]])
        self_trans = self.node_transform.build(E)

        # self_trans = tf.Print(self_trans, [self_trans], message='self_trans: ', summarize=100000000)

        def message_pass(adj, agg):
            sink_mask = (np.sum(adj, axis=-1) > 0)
            # sink_mask = np.float32(sink_mask)
            # sink_mask = np.float16(sink_mask)
            # adj = np.float16(adj)
            sink_mask = tf.cast(sink_mask, self.dtype)
            adj = tf.cast(adj, self.dtype)

            x = self_trans
            for i in range(self.k):
                # x = tf.Print(x, [x], message='pre agg: x', summarize=1000)
                x = agg.build(x, mask=adj)
                # x = tf.Print(x, [x], message='x', summarize=1000)
                x = sink_mask * tf.transpose(x)
                x = tf.transpose(x)
                x += self_trans

            return x

        def f(adj):
            n = adj.shape[0]
            t = np.zeros([bs * n] * 2, dtype=np.float32)
            for i in range(bs):
                t[i * n: (i + 1) * n, i * n: (i + 1) * n] = adj

            return t

        f_adj = f(f_adj)
        b_adj = f(b_adj)

        with tf.variable_scope('Forward_pass'):
            out_fpa = message_pass(f_adj, self.fpa)
        with tf.variable_scope('Backward_pass'):
            out_bpa = message_pass(b_adj, self.bpa)

        out = tf.concat([out_fpa, out_bpa], axis=-1)
        out = tf.cast(out, tf.float32)

        return out

    def mess_build(self, G, node_order, E):
        return Messenger.build(self, G, node_order, E)
