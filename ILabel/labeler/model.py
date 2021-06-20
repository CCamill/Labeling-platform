# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 15:33:59 2018
preprocessing data
@author: yang
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
from collections import OrderedDict
from tensorflow.contrib.layers import instance_norm
from tensorflow.contrib.layers.python.layers import initializers


def create_UNet(x, features_root, n_classes, dim):

    if dim == 2:
        net = create_2D_UNet(x, features_root, n_classes)
    elif dim == 3:
        net = create_3D_UNet(x, features_root, n_classes)
    else:
        raise ValueError("wrong dimension selected in configs.")

    return net['out_map']


def leaky_relu(x):
    """
    from https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/activation.py
    """
    half_alpha = 0.01
    return (0.5 + half_alpha) * x + (0.5 - half_alpha) * abs(x)


def create_2D_UNet(x, features_root, n_classes):

    net = OrderedDict()
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                   weights_initializer = initializers.variance_scaling_initializer(
                       factor=2.0, mode='FAN_IN', uniform=False), activation_fn=leaky_relu):

        net['encode/conv1_1'] = instance_norm(slim.conv2d(x, features_root, [3, 3]))
        net['encode/conv1_2'] = instance_norm(slim.conv2d(net['encode/conv1_1'], features_root, [3, 3]))
        net['encode/pool1'] = slim.max_pool2d(net['encode/conv1_2'], [2, 2])

        net['encode/conv2_1'] = instance_norm(slim.conv2d(net['encode/pool1'], features_root*2, [3, 3]))
        net['encode/conv2_2'] = instance_norm(slim.conv2d(net['encode/conv2_1'], features_root*2, [3, 3]))
        net['encode/pool2'] = slim.max_pool2d(net['encode/conv2_2'], [2, 2])

        net['encode/conv3_1'] = instance_norm(slim.conv2d(net['encode/pool2'], features_root*4, [3, 3]))
        net['encode/conv3_2'] = instance_norm(slim.conv2d(net['encode/conv3_1'], features_root*4, [3, 3]))
        net['encode/pool3'] = slim.max_pool2d(net['encode/conv3_2'], [2, 2])

        net['encode/conv4_1'] = instance_norm(slim.conv2d(net['encode/pool3'], features_root*8, [3, 3]))
        net['encode/conv4_2'] = instance_norm(slim.conv2d(net['encode/conv4_1'], features_root*8, [3, 3]))
        net['encode/pool4'] = slim.max_pool2d(net['encode/conv4_2'], [2, 2])

        net['encode/conv5_1'] = instance_norm(slim.conv2d(net['encode/pool4'], features_root*16, [3, 3]))
        net['encode/conv5_2'] = instance_norm(slim.conv2d(net['encode/conv5_1'], features_root*16, [3, 3]))

        net['decode/up_conv1'] = slim.conv2d_transpose(net['encode/conv5_2'], features_root * 8, 2,
                                                       stride=2, activation_fn=None, padding='VALID')
        net['decode/concat_c4_u1'] = tf.concat([net['encode/conv4_2'], net['decode/up_conv1']], 3)
        net['decode/conv1_1'] = instance_norm(slim.conv2d(net['decode/concat_c4_u1'], features_root * 8, [3, 3]))
        net['decode/conv1_2'] = instance_norm(slim.conv2d(net['decode/conv1_1'], features_root * 8, [3, 3]))

        net['decode/up_conv2'] = slim.conv2d_transpose(net['decode/conv1_2'], features_root * 4, 2,
                                                       stride=2, activation_fn=None, padding='VALID')
        net['decode/concat_c3_u2'] = tf.concat([net['encode/conv3_2'], net['decode/up_conv2']], 3)
        net['decode/conv2_1'] = instance_norm(slim.conv2d(net['decode/concat_c3_u2'], features_root * 4, [3, 3]))
        net['decode/conv2_2'] = instance_norm(slim.conv2d(net['decode/conv2_1'], features_root * 4, [3, 3]))

        net['decode/up_conv3'] = slim.conv2d_transpose(net['decode/conv2_2'], features_root * 2, 2,
                                                       stride=2, activation_fn=None, padding='VALID')
        net['decode/concat_c2_u3'] = tf.concat([net['encode/conv2_2'], net['decode/up_conv3']], 3)
        net['decode/conv3_1'] = instance_norm(slim.conv2d(net['decode/concat_c2_u3'], features_root * 2, [3, 3]))
        net['decode/conv3_2'] = instance_norm(slim.conv2d(net['decode/conv3_1'], features_root * 2, [3, 3]))

        net['decode/up_conv4'] = slim.conv2d_transpose(net['decode/conv3_2'], features_root, 2,
                                                       stride=2, activation_fn=None, padding='VALID')
        net['decode/concat_c1_u4'] = tf.concat([net['encode/conv1_2'], net['decode/up_conv4']], 3)
        net['decode/conv4_1'] = instance_norm(slim.conv2d(net['decode/concat_c1_u4'], features_root, [3, 3]))
        net['decode/conv4_2'] = instance_norm(slim.conv2d(net['decode/conv4_1'], features_root, [3, 3]))

        net['out_map'] = instance_norm(slim.conv2d(net['decode/conv4_2'], n_classes, [1, 1], activation_fn=None))

    return net


def create_3D_UNet(x, features_root=16, n_classes=2):

    net = OrderedDict()
    with slim.arg_scope([slim.conv3d, slim.conv3d_transpose],
                   weights_initializer = initializers.variance_scaling_initializer(
                       factor=2.0, mode='FAN_IN', uniform=False), activation_fn=leaky_relu):

        net['encode/conv1_1'] = instance_norm(slim.conv3d(x, features_root, [3, 3, 3]))
        net['encode/conv1_2'] = instance_norm(slim.conv3d(net['encode/conv1_1'], features_root, [3, 3, 3]))
        net['encode/pool1'] = slim.max_pool3d(net['encode/conv1_2'], kernel_size=[1, 2, 2], stride=[1,2,2])

        net['encode/conv2_1'] = instance_norm(slim.conv3d(net['encode/pool1'], features_root*2, [3, 3, 3]))
        net['encode/conv2_2'] = instance_norm(slim.conv3d(net['encode/conv2_1'], features_root*2, [3, 3, 3]))
        net['encode/pool2'] = slim.max_pool3d(net['encode/conv2_2'], kernel_size=[2, 2, 2], stride=[2,2,2])

        net['encode/conv3_1'] = instance_norm(slim.conv3d(net['encode/pool2'], features_root*4, [3, 3, 3]))
        net['encode/conv3_2'] = instance_norm(slim.conv3d(net['encode/conv3_1'], features_root*4, [3, 3, 3]))
        net['encode/pool3'] = slim.max_pool3d(net['encode/conv3_2'], [2, 2, 2])

        net['encode/conv4_1'] = instance_norm(slim.conv3d(net['encode/pool3'], features_root*8, [3, 3, 3]))
        net['encode/conv4_2'] = instance_norm(slim.conv3d(net['encode/conv4_1'], features_root*8, [3, 3, 3]))
        net['encode/pool4'] = slim.max_pool3d(net['encode/conv4_2'], [2, 2, 2])

        net['encode/conv5_1'] = instance_norm(slim.conv3d(net['encode/pool4'], features_root*16, [3, 3, 3]))
        net['encode/conv5_2'] = instance_norm(slim.conv3d(net['encode/conv5_1'], features_root*16, [3, 3, 3]))

        net['decode/up_conv1'] = slim.conv3d_transpose(net['encode/conv5_2'], features_root * 8, [2, 2, 2],
                                                       stride=2, activation_fn=None, padding='VALID', biases_initializer=None)
        net['decode/concat_c4_u1'] = tf.concat([net['encode/conv4_2'], net['decode/up_conv1']], 4)
        net['decode/conv1_1'] = instance_norm(slim.conv3d(net['decode/concat_c4_u1'], features_root * 8, [3, 3, 3]))
        net['decode/conv1_2'] = instance_norm(slim.conv3d(net['decode/conv1_1'], features_root * 8, [3, 3, 3]))

        net['decode/up_conv2'] = slim.conv3d_transpose(net['decode/conv1_2'], features_root * 4, [2, 2, 2],
                                                       stride=2, activation_fn=None, padding='VALID', biases_initializer=None)

        net['decode/concat_c3_u2'] = tf.concat([net['encode/conv3_2'], net['decode/up_conv2']], 4)
        net['decode/conv2_1'] = instance_norm(slim.conv3d(net['decode/concat_c3_u2'], features_root * 4, [3, 3, 3]))
        net['decode/conv2_2'] = instance_norm(slim.conv3d(net['decode/conv2_1'], features_root * 4, [3, 3, 3]))

        net['decode/up_conv3'] = slim.conv3d_transpose(net['decode/conv2_2'], features_root * 2, kernel_size=[2, 2, 2], stride=[2,2,2],
                                                       activation_fn=None, padding='VALID', biases_initializer=None)
        net['decode/concat_c2_u3'] = tf.concat([net['encode/conv2_2'], net['decode/up_conv3']], 4)
        net['decode/conv3_1'] = instance_norm(slim.conv3d(net['decode/concat_c2_u3'], features_root * 2, [3, 3, 3]))
        net['decode/conv3_2'] = instance_norm(slim.conv3d(net['decode/conv3_1'], features_root * 2, [3, 3, 3]))

        net['decode/up_conv4'] = slim.conv3d_transpose(net['decode/conv3_2'], features_root,  [1, 2, 2],
                                                       stride=[1, 2, 2], activation_fn=None, padding='VALID', biases_initializer=None)

        net['decode/concat_c1_u4'] = tf.concat([net['encode/conv1_2'], net['decode/up_conv4']], 4)
        net['decode/conv4_1'] = instance_norm(slim.conv3d(net['decode/concat_c1_u4'], features_root, [3, 3, 3]))
        net['decode/conv4_2'] = instance_norm(slim.conv3d(net['decode/conv4_1'], features_root, [3, 3, 3]))

        net['out_map'] = instance_norm(slim.conv3d(net['decode/conv4_2'], n_classes, [1, 1, 1], activation_fn=None))

    return net

def create_VNet(x, features_root, n_classes, dim):

    if dim == 2:
        net = create_2D_VNet(x, features_root, n_classes)
    elif dim == 3:
        net = create_3D_VNet(x, features_root, n_classes)
    else:
        raise ValueError("wrong dimension selected in configs.")

    return net['out_map']
    
def create_2D_VNet(x, features_root=16, n_classes=2):

    net = OrderedDict()
    with slim.arg_scope([slim.conv2d, slim.conv3d_transpose],weights_initializer = initializers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False), activation_fn=leaky_relu):
        ######## DSN_VNET for the 256x256 to 256x256 ###################################
        # input MR 256x256x1
        # level 0
        # convolution component 0
        net['encode/conv0_1'] = instance_norm(slim.conv2d(x, features_root, [5, 5]))
        # encode/conv0_1 out_put 256x256xfeature_root
        
        # level 1
        # convolution component 1
        net['encode/conv1_1'] = instance_norm(slim.conv2d(net['encode/conv0_1'], features_root, [5, 5]))
        # encode/conv1_1 out_put 256x256xfeature_root
        net['encode/conv1_2'] = instance_norm(slim.conv2d(net['encode/conv1_1'], features_root, [5, 5]))
        # encode/conv1_1 out_put 256x256xfeature_root
        # down comvolution 1
        net['encode/pool1'] = slim.max_pool2d(net['encode/conv1_2'], kernel_size=[2, 2], stride=[2, 2])
        # encode/pool1 out_put 128x128xfeature_root
        
        # level 2
        # convolution component 2
        net['encode/conv2_1'] = instance_norm(slim.conv2d(net['encode/pool1'], features_root*2, [5, 5]))
        # encode/conv2_1 out_put 128x128xfeature_root*2
        net['encode/conv2_2'] = instance_norm(slim.conv2d(net['encode/conv2_1'], features_root*2, [5, 5]))
        # encode/conv2_2 out_put 128x128xfeature_root*2
        # down comvolution 2
        net['encode/pool2'] = slim.max_pool2d(net['encode/conv2_2'], kernel_size=[2, 2], stride=[2 ,2])
        # encode/pool2 out_put 64x64xfeature_root*2
        
        # level 3
        # convolution component 3
        net['encode/conv3_1'] = instance_norm(slim.conv2d(net['encode/pool2'], features_root*4, [5, 5]))
        # encode/conv3_1 out_put 64x64xfeature_root*4
        net['encode/conv3_2'] = instance_norm(slim.conv2d(net['encode/conv3_1'], features_root*4, [5, 5]))
        # encode/conv3_2 out_put 64x64xfeature_root*4
        # down comvolution 3
        net['encode/pool3'] = slim.max_pool2d(net['encode/conv3_2'], kernel_size=[2, 2], stride=[2, 2])
        # encode/pool3 out_put 32x32xfeature_root*4
        
        # level 4
        # convolution component 4
        net['encode/conv4_1'] = instance_norm(slim.conv2d(net['encode/pool3'], features_root*8, [3, 3]))
        # encode/conv4_1 out_put 32x32xfeature_root*8
        net['encode/conv4_2'] = instance_norm(slim.conv2d(net['encode/conv4_1'], features_root*8, [3, 3]))
        # encode/conv4_2 out_put 32x32xfeature_root*8
        # down comvolution 4
        net['encode/pool4'] = slim.max_pool2d(net['encode/conv4_2'], kernel_size=[2, 2], stride=[2, 2])
        # encode/pool4 out_put 16x16xfeature_root*8
        
        # level 5
        # convolution component 5
        net['encode/conv5_1'] = instance_norm(slim.conv2d(net['encode/pool4'], features_root*16, [2, 2]))
        # encode/conv5_1 out_put 16x16xfeature_root*16
        net['encode/conv5_2'] = instance_norm(slim.conv2d(net['encode/conv5_1'], features_root*16, [2, 2]))
        # encode/conv5_2 out_put 16x16xfeature_root*16
        # de-convolution 5
        net['decode/up_conv5'] = slim.conv2d_transpose(net['encode/conv5_2'], features_root*8, [2, 2], stride=2, activation_fn=None, padding='SAME', biases_initializer=None)
        # decode/up_conv5 out_put 32x32xfeature_root*8
              
        
        ######## expanding path ########################################################
        # level 4+
        # concatenante component 4
        net['decode/concat_4'] = tf.concat([net['encode/conv4_2'], net['decode/up_conv5']], 3)
        # decode/concat_4 out_put 32x32xfeature_root*16
        # convolution component 4+(6)
        net['decode/conv6_1'] = instance_norm(slim.conv2d(net['decode/concat_4'], features_root*8, [3, 3]))
        # encode/conv6_1 out_put 32x32xfeature_root*8
        net['decode/conv6_2'] = instance_norm(slim.conv2d(net['decode/conv6_1'], features_root*8, [3, 3]))
        # encode/conv6_2 out_put 32x32xfeature_root*8
        # de_convolution component 4
        net['decode/up_conv4_0'] = slim.conv2d_transpose(net['decode/conv6_2'], features_root*4, [3, 3], stride=2, activation_fn=None, padding='SAME', biases_initializer=None)
        # decode/up_conv4_0 out_put 64x64xfeature_root*4
        
        
        # level 3+
        # concatenante component 3
        net['decode/concat_3'] = tf.concat([net['encode/conv3_2'], net['decode/up_conv4_0']], 3)
        # decode/concat_3 out_put 64x64xfeature_root*8
        # convolution component 3+(7)
        net['decode/conv7_1'] = instance_norm(slim.conv2d(net['decode/concat_3'], features_root*4, [5, 5]))
        # encode/conv7_1 out_put 64x64xfeature_root*4
        net['decode/conv7_2'] = instance_norm(slim.conv2d(net['decode/conv7_1'], features_root*4, [5, 5]))
        # encode/conv7_2 out_put 64x64xfeature_root*4
        # de_convolution component 3
        net['decode/up_conv3_0'] = slim.conv2d_transpose(net['decode/conv7_2'], features_root*2, [2, 2], stride=2, activation_fn=None, padding='SAME', biases_initializer=None)
        # decode/up_conv3_0 out_put 128x128xfeature_root*2
        
        
        # level 2+
        # concatenante component 2
        net['decode/concat_2'] = tf.concat([net['encode/conv2_2'], net['decode/up_conv3_0']], 3)
        # decode/concat_2 out_put 128x128xfeature_root*4
        # convolution component 2+(8)
        net['decode/conv8_1'] = instance_norm(slim.conv2d(net['decode/concat_2'], features_root*2, [5, 5]))
        # encode/conv8_1 out_put 128x128xfeature_root*2
        net['decode/conv8_2'] = instance_norm(slim.conv2d(net['decode/conv8_1'], features_root*2, [5, 5]))
        # encode/conv8_2 out_put 128x128xfeature_root*2
        # de_convolution component 2
        net['decode/up_conv2_0'] = slim.conv2d_transpose(net['decode/conv8_2'], features_root, [2, 2], stride=2, activation_fn=None, padding='SAME', biases_initializer=None)
        # decode/up_conv2_0 out_put 256x256xfeature_root

        # level 1+
        # concatenante component 1
        net['decode/concat_1'] = tf.concat([net['encode/conv1_2'], net['decode/up_conv2_0']], 3)
        # decode/concat_1 out_put 256x256xfeature_root*2
        # convolution component 1+(9)
        net['decode/conv9_1'] = instance_norm(slim.conv2d(net['decode/concat_1'], features_root, [5, 5]))
        # encode/conv9_1 out_put 256x256xfeature_root
        net['decode/conv9_2'] = instance_norm(slim.conv2d(net['decode/conv9_1'], features_root, [5, 5]))
        # encode/conv9_2 out_put 256x256xfeature_root
        
        # level 0+
        # convolutional component 0+(10)
        net['out_map'] = instance_norm(slim.conv2d(net['decode/conv9_2'], n_classes, [1, 1], activation_fn=None))
        
    return net

def create_3D_VNet(x, features_root=16, n_classes=2):

    net = OrderedDict()
    with slim.arg_scope([slim.conv3d, slim.conv3d_transpose],weights_initializer = initializers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False), activation_fn=leaky_relu):
        ######## DSN_VNET for the 32X32X32 to 32x32x32 ###################################
        # input MR 32x32x32x1
        # level 0
        # convolution component 0
        net['encode/conv0_1'] = instance_norm(slim.conv3d(x, features_root, [5, 5, 5]))
        # encode/conv0_1 out_put 32x32x32x16
        
        # level 1
        # convolution component 1
        net['encode/conv1_1'] = instance_norm(slim.conv3d(net['encode/conv0_1'], features_root, [5, 5, 5]))
        # encode/conv1_1 out_put 32x32x32x16
        net['encode/conv1_2'] = instance_norm(slim.conv3d(net['encode/conv1_1'], features_root, [5, 5, 5]))
        # encode/conv1_1 out_put 32x32x32x16
        # down comvolution 1
        net['encode/pool1'] = slim.max_pool3d(net['encode/conv1_2'], kernel_size=[2, 2, 2], stride=[2,2,2])
        # encode/pool1 out_put 16x16x16x16
        
        # level 2
        # convolution component 2
        net['encode/conv2_1'] = instance_norm(slim.conv3d(net['encode/pool1'], features_root*2, [5, 5, 5]))
        # encode/conv2_1 out_put 16x16x16x32
        net['encode/conv2_2'] = instance_norm(slim.conv3d(net['encode/conv2_1'], features_root*2, [5, 5, 5]))
        # encode/conv2_2 out_put 16x16x16x32
        # down comvolution 2
        net['encode/pool2'] = slim.max_pool3d(net['encode/conv2_2'], kernel_size=[2, 2, 2], stride=[2,2,2])
        # encode/pool2 out_put 8x8x8x32
        
        # level 3
        # convolution component 3
        net['encode/conv3_1'] = instance_norm(slim.conv3d(net['encode/pool2'], features_root*4, [5, 5, 5]))
        # encode/conv3_1 out_put 8x8x8x64
        net['encode/conv3_2'] = instance_norm(slim.conv3d(net['encode/conv3_1'], features_root*4, [5, 5, 5]))
        # encode/conv3_2 out_put 8x8x8x64
        # down comvolution 3
        net['encode/pool3'] = slim.max_pool3d(net['encode/conv3_2'], kernel_size=[2, 2, 2], stride=[2,2,2])
        # encode/pool3 out_put 4x4x4x64
        
        # level 4
        # convolution component 4
        net['encode/conv4_1'] = instance_norm(slim.conv3d(net['encode/pool3'], features_root*8, [3, 3, 3]))
        # encode/conv4_1 out_put 4x4x4x128
        net['encode/conv4_2'] = instance_norm(slim.conv3d(net['encode/conv4_1'], features_root*8, [3, 3, 3]))
        # encode/conv4_2 out_put 4x4x4x128
        # down comvolution 4
        net['encode/pool4'] = slim.max_pool3d(net['encode/conv4_2'], kernel_size=[2, 2, 2], stride=[2,2,2])
        # encode/pool4 out_put 2x2x2x128
        
        # level 5
        # convolution component 5
        net['encode/conv5_1'] = instance_norm(slim.conv3d(net['encode/pool4'], features_root*16, [2, 2, 2]))
        # encode/conv5_1 out_put 2x2x2x256
        net['encode/conv5_2'] = instance_norm(slim.conv3d(net['encode/conv5_1'], features_root*16, [2, 2, 2]))
        # encode/conv5_2 out_put 2x2x2x256
        # de-convolution 5
        net['decode/up_conv5'] = slim.conv3d_transpose(net['encode/conv5_2'], features_root*8, [2, 2, 2], stride=2, activation_fn=None, padding='SAME', biases_initializer=None)
        # decode/up_conv5 out_put 4x4x4x128
        
        ######## expanding path ########################################################
        # level 4+
        # concatenante component 4
        net['decode/concat_4'] = tf.concat([net['encode/conv4_2'], net['decode/up_conv5']], 4)
        # decode/concat_4 out_put 4x4x4x256
        # convolution component 4+(6)
        net['decode/conv6_1'] = instance_norm(slim.conv3d(net['decode/concat_4'], features_root*8, [3, 3, 3]))
        # encode/conv6_1 out_put 4x4x4x128
        net['decode/conv6_2'] = instance_norm(slim.conv3d(net['decode/conv6_1'], features_root*8, [3, 3, 3]))
        # encode/conv6_2 out_put 4x4x4x128
        # de_convolution component 4
        net['decode/up_conv4_0'] = slim.conv3d_transpose(net['decode/conv6_2'], features_root*4, [3, 3, 3], stride=2, activation_fn=None, padding='SAME', biases_initializer=None)
        # decode/up_conv4_0 out_put 8x8x8x64
       
        
        # level 3+
        # concatenante component 3
        net['decode/concat_3'] = tf.concat([net['encode/conv3_2'], net['decode/up_conv4_0']], 4)
        # decode/concat_3 out_put 8x8x8x128
        # convolution component 3+(7)
        net['decode/conv7_1'] = instance_norm(slim.conv3d(net['decode/concat_3'], features_root*4, [5, 5, 5]))
        # encode/conv7_1 out_put 8x8x8x64
        net['decode/conv7_2'] = instance_norm(slim.conv3d(net['decode/conv7_1'], features_root*4, [5, 5, 5]))
        # encode/conv7_2 out_put 8x8x8x64
        # de_convolution component 3
        net['decode/up_conv3_0'] = slim.conv3d_transpose(net['decode/conv7_2'], features_root*2, [2, 2, 2], stride=2, activation_fn=None, padding='SAME', biases_initializer=None)
        # decode/up_conv3_0 out_put 16x16x16x32
        
        
        # level 2+
        # concatenante component 2
        net['decode/concat_2'] = tf.concat([net['encode/conv2_2'], net['decode/up_conv3_0']], 4)
        # decode/concat_2 out_put 16x16x16x64
        # convolution component 2+(8)
        net['decode/conv8_1'] = instance_norm(slim.conv3d(net['decode/concat_2'], features_root*2, [5, 5, 5]))
        # encode/conv8_1 out_put 16x16x16x32
        net['decode/conv8_2'] = instance_norm(slim.conv3d(net['decode/conv8_1'], features_root*2, [5, 5, 5]))
        # encode/conv8_2 out_put 16x16x16x32
        # de_convolution component 2
        net['decode/up_conv2_0'] = slim.conv3d_transpose(net['decode/conv8_2'], features_root, [2, 2, 2], stride=2, activation_fn=None, padding='SAME', biases_initializer=None)
        # decode/up_conv2_0 out_put 32x32x32x16

        # level 1+
        # concatenante component 1
        net['decode/concat_1'] = tf.concat([net['encode/conv1_2'], net['decode/up_conv2_0']], 4)
        # decode/concat_1 out_put 32x32x32x32
        # convolution component 1+(9)
        net['decode/conv9_1'] = instance_norm(slim.conv3d(net['decode/concat_1'], features_root, [5, 5, 5]))
        # encode/conv9_1 out_put 32x32x32x16
        net['decode/conv9_2'] = instance_norm(slim.conv3d(net['decode/conv9_1'], features_root, [5, 5, 5]))
        # encode/conv9_2 out_put 32x32x32x16
        
        # level 0+
        # convolutional component 0+(10)
        net['out_map'] = instance_norm(slim.conv3d(net['decode/conv9_2'], n_classes, [1, 1, 1], activation_fn=None))
    return net