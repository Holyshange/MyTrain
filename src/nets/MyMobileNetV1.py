from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

def inference(inputs, num_classes=1000, keep_prob=0.8, weight_decay=0.0, is_training=True):
    normalizer_params = {
        'decay': 0.995,
        'epsilon': 0.001,
        'updates_collections': None,
        'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
        }
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d], 
                        activation_fn=tf.nn.relu, 
                        normalizer_fn=slim.batch_norm, 
                        normalizer_params=normalizer_params, 
                        weights_initializer=slim.initializers.xavier_initializer(), 
                        weights_regularizer=slim.l2_regularizer(weight_decay), 
                        padding='SAME'):
        return MyMobileNetV1(inputs, 
                             num_classes=num_classes, 
                             keep_prob=keep_prob, 
                             weight_decay=weight_decay, 
                             is_training=is_training)

def MyMobileNetV1(inputs, num_classes=1000, keep_prob=0.8, weight_decay=0.0, 
                  is_training=True, scope='MobileNetV1'):
    end_points = {}
    with tf.variable_scope(scope, [inputs]):
        with slim.arg_scope([slim.batch_norm], 
                            activation_fn=None, 
                            is_training=is_training):
            net = slim.conv2d(inputs, 32, 3, stride=2, scope='Conv_1')
            end_points['Conv_1'] = net
            net = _depthwise_separable_conv2d(net, 64, "DS_Conv_2")
            end_points['DS_Conv_2'] = net
            net = _depthwise_separable_conv2d(net, 128, "DS_Conv_3", downsample=True)
            end_points['DS_Conv_3'] = net
            net = _depthwise_separable_conv2d(net, 128, "DS_Conv_4")
            end_points['DS_Conv_4'] = net
            net = _depthwise_separable_conv2d(net, 256, "DS_Conv_5", downsample=True)
            end_points['DS_Conv_5'] = net
            net = _depthwise_separable_conv2d(net, 256, "DS_Conv_6")
            end_points['DS_Conv_6'] = net
            net = _depthwise_separable_conv2d(net, 512, "DS_Conv_7", downsample=True)
            end_points['DS_Conv_7'] = net
            net = _depthwise_separable_conv2d(net, 512, "DS_Conv_8")
            end_points['DS_Conv_8'] = net
            net = _depthwise_separable_conv2d(net, 512, "DS_Conv_9")
            end_points['DS_Conv_9'] = net
            net = _depthwise_separable_conv2d(net, 512, "DS_Conv_10")
            end_points['DS_Conv_10'] = net
            net = _depthwise_separable_conv2d(net, 512, "DS_Conv_11")
            end_points['DS_Conv_11'] = net
            net = _depthwise_separable_conv2d(net, 512, "DS_Conv_12")
            end_points['DS_Conv_12'] = net
            net = _depthwise_separable_conv2d(net, 1024, "DS_Conv_13", downsample=True)
            end_points['DS_Conv_13'] = net
            net = _depthwise_separable_conv2d(net, 1024, "DS_Conv_14")
            end_points['DS_Conv_14'] = net
            net = slim.avg_pool2d(net, [7, 7], padding='VALID', scope='AVG_Pool_15')
            end_points['AVG_Pool_15'] = net
            net = slim.flatten(net)
            net = slim.dropout(net, keep_prob, is_training=is_training, scope='Dropout')
            logits = slim.fully_connected(net, num_classes, 
                                          activation_fn=None, 
                                          weights_initializer=slim.initializers.xavier_initializer(), 
                                          weights_regularizer=slim.l2_regularizer(weight_decay), 
                                          scope='FC_16', reuse=False)
            end_points['Logits'] = logits
            predictions = slim.softmax(logits, scope='Predictions')
            end_points['Predictions'] = predictions
            return logits, end_points

def _depthwise_separable_conv2d(inputs, num_filters, scope, downsample=False):
    stride = 2 if downsample else 1
    inputs_shape = inputs.get_shape().as_list()
    in_channels = inputs_shape[-1]
    with tf.variable_scope(scope):
        net = slim.separable_conv2d(inputs, in_channels, 3, 1, stride=stride, 
                                    scope='DW_CONV')
        net = slim.conv2d(net, num_filters, 1, stride=1, scope='PW_CONV')
        return net





