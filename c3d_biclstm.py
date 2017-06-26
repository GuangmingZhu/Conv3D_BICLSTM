import io
import sys
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import tensorlayer as tl
from tensorflow.python.framework import ops
import ConvLSTMCell as clstm

def c3d_biclstm(inputs, num_classes, reuse, is_training):
  with tf.device('/gpu:0'):
    with tf.variable_scope('Conv3D_ConvLSTM', reuse=reuse):
      tl.layers.set_name_reuse(reuse)
      if inputs.get_shape().ndims!=5:
        raise Exception("The input dimension of 3DCNN must be rank 5")
      network_input = tl.layers.InputLayer(inputs, name='input_layer')      #Input Layer
      # 3DCNN-BN Layer 1
      conv3d_1 = tl.layers.Conv3dLayer(network_input,
                                        act=tf.identity,
                                        shape=[3,3,3,3,64],
                                        strides=[1,1,1,1,1],
                                        padding='SAME',
                                        name='Conv3d_1')
      conv3d_1 = tl.layers.BatchNormLayer(layer=conv3d_1, 
                                        act=tf.nn.relu,
                                        is_train=is_training,
                                        name='BatchNorm_1')
      pool3d_1 = tl.layers.PoolLayer(conv3d_1,
                                        ksize=[1,1,2,2,1],
                                        strides=[1,1,2,2,1],
                                        padding='SAME',
                                        pool = tf.nn.max_pool3d,
                                        name='Pool3D_1')
      # 3DCNN-Incep Layer 2
      conv3d_2_3x3 = tl.layers.Conv3dLayer(pool3d_1, 
                                        act=tf.identity, 
                                        shape=[3,3,3,64,128], 
                                        strides=[1,1,1,1,1],
                                        padding='SAME',
                                        name='Conv3d_2_3x3')
      conv3d_2_3x3 = tl.layers.BatchNormLayer(layer=conv3d_2_3x3, 
                                        act=tf.nn.relu,
                                        is_train=is_training, 
                                        name='BatchNorm_2_3x3')
      pool3d_2 = tl.layers.PoolLayer(conv3d_2_3x3,
                                        ksize=[1,2,2,2,1],
                                        strides=[1,2,2,2,1],
                                        padding='SAME',
                                        pool = tf.nn.max_pool3d,
                                        name='Pool3D_2')
      # 3DCNN-Resnet Layer 1
      conv3d_3a_3x3 = tl.layers.Conv3dLayer(pool3d_2, 
                                        act=tf.identity, 
                                        shape=[3,3,3,128,256],
                                        strides=[1,1,1,1,1],
                                        padding='SAME',
                                        name='Conv3d_3a_3x3')
      conv3d_3b_3x3 = tl.layers.Conv3dLayer(conv3d_3a_3x3, 
                                        act=tf.identity, 
                                        shape=[3,3,3,256,256],
                                        strides=[1,1,1,1,1],
                                        padding='SAME',
                                        name='Conv3d_3b_3x3')
      conv3d_3_3x3 = tl.layers.BatchNormLayer(layer=conv3d_3b_3x3, 
                                        act=tf.nn.relu,
                                        is_train=is_training, 
                                        name='BatchNorm_3_3x3')
      # ConvLstm Layer
      shape3d = conv3d_3_3x3.outputs.get_shape().as_list()
      num_steps = shape3d[1]
      fwconvlstm1 = tl.layers.RNNLayer(conv3d_3_3x3,
                                        cell_fn=clstm.ConvLSTMCell,
                                        cell_init_args={'state_is_tuple':False},
                                        n_hidden=256,
                                        initializer=tf.random_uniform_initializer(-0.1, 0.1),
                                        n_steps=num_steps,
                                        return_last=False,
                                        return_seq_2d=False,
                                        name='fwclstm_layer_1')
      fwconvlstm2 = tl.layers.RNNLayer(fwconvlstm1,
                                        cell_fn=clstm.ConvLSTMCell,
                                        cell_init_args={'state_is_tuple':False},
                                        n_hidden=384,
                                        initializer=tf.random_uniform_initializer(-0.1, 0.1),
                                        n_steps=num_steps,
                                        return_last=False,
                                        return_seq_2d=False,
                                        name='fwclstm_layer_2')
      bwconvlstm1 = tl.layers.RNNLayer(conv3d_3_3x3,
                                        cell_fn=clstm.ConvLSTMCell,
                                        cell_init_args={'state_is_tuple':False},
                                        n_hidden=256,
                                        initializer=tf.random_uniform_initializer(-0.1, 0.1),
                                        n_steps=num_steps,
                                        return_last=False,
                                        return_seq_2d=False,
                                        reverse=True,
                                        name='bwclstm_layer_1')
      bwconvlstm2 = tl.layers.RNNLayer(bwconvlstm1,
                                        cell_fn=clstm.ConvLSTMCell,
                                        cell_init_args={'state_is_tuple':False},
                                        n_hidden=384,
                                        initializer=tf.random_uniform_initializer(-0.1, 0.1),
                                        n_steps=num_steps,
                                        return_last=False,
                                        return_seq_2d=False,
                                        name='bwclstm_layer_2')
      convlstm2 = tl.layers.ConcatLayer(layer=[fwconvlstm2,bwconvlstm2],
                                        concat_dim=4,
                                        name='fwbw_concat')
      convlstm2 = tl.layers.ReshapeLayer(layer=convlstm2,
                                        shape=[-1, shape3d[2], shape3d[3], 768],
                                        name='clstm_reshape')
      # Conv2d
      conv2d_1 = tl.layers.Conv2dLayer(convlstm2,
                                        act=tf.identity, 
                                        shape=[3,3,768,128],
                                        strides=[1,1,1,1],
                                        padding='SAME',
                                        name='Conv2d_1')
      conv2d_1 = tl.layers.BatchNormLayer(layer=conv2d_1, 
                                        act=tf.nn.relu,
                                        is_train=is_training,
                                        name='BatchNorm_3')
      conv2d_1 = tl.layers.PoolLayer(conv2d_1,
                                        ksize=[1,2,2,1],
                                        strides=[1,2,2,1],
                                        padding='SAME',
                                        pool = tf.nn.max_pool,
                                        name='Conv2d_Pool_1')
      conv2d_2 = tl.layers.Conv2dLayer(conv2d_1,
                                        act=tf.identity, 
                                        shape=[3,3,128,256],
                                        strides=[1,1,1,1],
                                        padding='SAME',
                                        name='Conv2d_2')
      conv2d_2 = tl.layers.BatchNormLayer(layer=conv2d_2, 
                                        act=tf.nn.relu,
                                        is_train=is_training,
                                        name='BatchNorm_4')
      conv2d_2 = tl.layers.PoolLayer(conv2d_2,
                                        ksize=[1,2,2,1],
                                        strides=[1,2,2,1],
                                        padding='SAME',
                                        pool = tf.nn.max_pool,
                                        name='Conv2d_Pool_2')
      conv2d_3 = tl.layers.Conv2dLayer(conv2d_2,
                                        act=tf.identity, 
                                        shape=[3,3,256,256],
                                        strides=[1,1,1,1],
                                        padding='SAME',
                                        name='Conv2d_3')
      conv2d_3 = tl.layers.BatchNormLayer(layer=conv2d_3, 
                                        act=tf.nn.relu,
                                        is_train=is_training,
                                        name='BatchNorm_5')
      conv2d_3 = tl.layers.PoolLayer(conv2d_3,
                                        ksize=[1,2,2,1],
                                        strides=[1,2,2,1],
                                        padding='SAME',
                                        pool = tf.nn.max_pool,
                                        name='Conv2d_Pool_3')
      # FC1
      shape2d = conv2d_3.outputs.get_shape().as_list()
      conv2d_2_3d = tl.layers.ReshapeLayer(layer=conv2d_3,
                                        shape=[-1, num_steps, shape2d[1], shape2d[2], shape2d[3]],
                                        name='conv2d_3_reshape')
      pooling_feats = tl.layers.PoolLayer(conv2d_2_3d,
                                        ksize=[1,num_steps,1,1,1],
                                        strides=[1,num_steps,1,1,1],
                                        padding='SAME',
                                        pool = tf.nn.avg_pool3d,
                                        name='Pooling_Feats')
      flatten_1 = tl.layers.FlattenLayer(pooling_feats, 
                                        name='Flatten_1')
      classes = tl.layers.DropconnectDenseLayer(flatten_1, 
                                        keep=0.5,
                                        n_units=num_classes,
                                        act=tf.identity,
                                        name='Classes')
    return flatten_1, classes
