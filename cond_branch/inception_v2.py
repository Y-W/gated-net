"""Contains common code shared by all inception models.
Usage of arg scope:
  with slim.arg_scope(inception_arg_scope()):
    logits, end_points = inception.inception_v3(images, num_classes,
                                                is_training=is_training)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops

slim = tf.contrib.slim

NUM_CLASSES=1001

WEIGHT_DECAY=0.0 # 0.00004
BATCH_NORM_DECAY=0.99 # 0.9997
BATCH_NORM_EPSILON=0.001

DROPOUT_KEEP=0.8

TRAIN_COMMON_SEG=False

BATCH_NORM_PARAMS = {
      # Decay for the moving averages.
      'decay': BATCH_NORM_DECAY,
      # epsilon to prevent 0s in variance.
      'epsilon': BATCH_NORM_EPSILON,
      # collection containing update_ops.
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
}


def inception_arg_scope():
  """Defines the default arg scope for inception models.
  Args:
    weight_decay: The weight decay to use for regularizing the model.
    use_batch_norm: "If `True`, batch_norm is applied after each convolution.
    batch_norm_decay: Decay for batch norm moving average.
    batch_norm_epsilon: Small float added to variance to avoid dividing by zero
      in batch norm.
  Returns:
    An `arg_scope` to use for the inception models.
  """
  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      weights_regularizer=slim.l2_regularizer(WEIGHT_DECAY)):
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=slim.variance_scaling_initializer(),
        activation_fn=tf.nn.relu,
        normalizer_fn=slim.batch_norm,
        normalizer_params=BATCH_NORM_PARAMS) as sc:
      return sc


trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


def inception_v2_common_seg(inputs, end_points, trainable=TRAIN_COMMON_SEG):
  # Used to find thinned depths for each layer.
  depth = lambda d: max(int(d), 16)
  with slim.arg_scope(
      [slim.conv2d, slim.max_pool2d, slim.avg_pool2d, slim.separable_conv2d],
      stride=1, padding='SAME'):

    # 224 x 224 x 3
    end_point = 'Conv2d_1a_7x7'
    depthwise_multiplier = min(int(depth(64) / 3), 8)
    net = slim.separable_conv2d(
        inputs, depth(64), [7, 7], depth_multiplier=depthwise_multiplier,
        stride=2, weights_initializer=trunc_normal(1.0),
        scope=end_point, trainable=trainable)
    end_points[end_point] = net

    # 112 x 112 x 64
    end_point = 'MaxPool_2a_3x3'
    net = slim.max_pool2d(net, [3, 3], scope=end_point, stride=2)
    end_points[end_point] = net

    # 56 x 56 x 64
    end_point = 'Conv2d_2b_1x1'
    net = slim.conv2d(net, depth(64), [1, 1], scope=end_point,
                      weights_initializer=trunc_normal(0.1), trainable=trainable)
    end_points[end_point] = net

    # 56 x 56 x 64
    end_point = 'Conv2d_2c_3x3'
    net = slim.conv2d(net, depth(192), [3, 3], scope=end_point, trainable=trainable)
    end_points[end_point] = net

    # 56 x 56 x 192
    end_point = 'MaxPool_3a_3x3'
    net = slim.max_pool2d(net, [3, 3], scope=end_point, stride=2)
    end_points[end_point] = net
    # 28 x 28 x 192
  return net, end_points


def inception_v2_branch_seg(inputs, end_points):
  depth = lambda d: max(int(d), 16)
  with slim.arg_scope(
      [slim.conv2d, slim.max_pool2d, slim.avg_pool2d, slim.separable_conv2d],
      stride=1, padding='SAME'):

    # 28 x 28 x 192
    # Inception module.
    end_point = 'Mixed_3b'
    with tf.variable_scope(end_point):
      with tf.variable_scope('Branch_0'):
        branch_0 = slim.conv2d(inputs, depth(64), [1, 1], scope='Conv2d_0a_1x1')
      with tf.variable_scope('Branch_1'):
        branch_1 = slim.conv2d(
            inputs, depth(64), [1, 1],
            weights_initializer=trunc_normal(0.09),
            scope='Conv2d_0a_1x1')
        branch_1 = slim.conv2d(branch_1, depth(64), [3, 3],
                                scope='Conv2d_0b_3x3')
      with tf.variable_scope('Branch_2'):
        branch_2 = slim.conv2d(
            inputs, depth(64), [1, 1],
            weights_initializer=trunc_normal(0.09),
            scope='Conv2d_0a_1x1')
        branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                scope='Conv2d_0b_3x3')
        branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                scope='Conv2d_0c_3x3')
      with tf.variable_scope('Branch_3'):
        branch_3 = slim.avg_pool2d(inputs, [3, 3], scope='AvgPool_0a_3x3')
        branch_3 = slim.conv2d(
            branch_3, depth(32), [1, 1],
            weights_initializer=trunc_normal(0.1),
            scope='Conv2d_0b_1x1')
      net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net

    # 28 x 28 x 256
    end_point = 'Mixed_3c'
    with tf.variable_scope(end_point):
      with tf.variable_scope('Branch_0'):
        branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
      with tf.variable_scope('Branch_1'):
        branch_1 = slim.conv2d(
            net, depth(64), [1, 1],
            weights_initializer=trunc_normal(0.09),
            scope='Conv2d_0a_1x1')
        branch_1 = slim.conv2d(branch_1, depth(96), [3, 3],
                                scope='Conv2d_0b_3x3')
      with tf.variable_scope('Branch_2'):
        branch_2 = slim.conv2d(
            net, depth(64), [1, 1],
            weights_initializer=trunc_normal(0.09),
            scope='Conv2d_0a_1x1')
        branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                scope='Conv2d_0b_3x3')
        branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                scope='Conv2d_0c_3x3')
      with tf.variable_scope('Branch_3'):
        branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
        branch_3 = slim.conv2d(
            branch_3, depth(64), [1, 1],
            weights_initializer=trunc_normal(0.1),
            scope='Conv2d_0b_1x1')
      net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net

    # 28 x 28 x 320
    end_point = 'Mixed_4a'
    with tf.variable_scope(end_point):
      with tf.variable_scope('Branch_0'):
        branch_0 = slim.conv2d(
            net, depth(128), [1, 1],
            weights_initializer=trunc_normal(0.09),
            scope='Conv2d_0a_1x1')
        branch_0 = slim.conv2d(branch_0, depth(160), [3, 3], stride=2,
                                scope='Conv2d_1a_3x3')
      with tf.variable_scope('Branch_1'):
        branch_1 = slim.conv2d(
            net, depth(64), [1, 1],
            weights_initializer=trunc_normal(0.09),
            scope='Conv2d_0a_1x1')
        branch_1 = slim.conv2d(
            branch_1, depth(96), [3, 3], scope='Conv2d_0b_3x3')
        branch_1 = slim.conv2d(
            branch_1, depth(96), [3, 3], stride=2, scope='Conv2d_1a_3x3')
      with tf.variable_scope('Branch_2'):
        branch_2 = slim.max_pool2d(
            net, [3, 3], stride=2, scope='MaxPool_1a_3x3')
      net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
      end_points[end_point] = net

    # 14 x 14 x 576
    end_point = 'Mixed_4b'
    with tf.variable_scope(end_point):
      with tf.variable_scope('Branch_0'):
        branch_0 = slim.conv2d(net, depth(224), [1, 1], scope='Conv2d_0a_1x1')
      with tf.variable_scope('Branch_1'):
        branch_1 = slim.conv2d(
            net, depth(64), [1, 1],
            weights_initializer=trunc_normal(0.09),
            scope='Conv2d_0a_1x1')
        branch_1 = slim.conv2d(
            branch_1, depth(96), [3, 3], scope='Conv2d_0b_3x3')
      with tf.variable_scope('Branch_2'):
        branch_2 = slim.conv2d(
            net, depth(96), [1, 1],
            weights_initializer=trunc_normal(0.09),
            scope='Conv2d_0a_1x1')
        branch_2 = slim.conv2d(branch_2, depth(128), [3, 3],
                                scope='Conv2d_0b_3x3')
        branch_2 = slim.conv2d(branch_2, depth(128), [3, 3],
                                scope='Conv2d_0c_3x3')
      with tf.variable_scope('Branch_3'):
        branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
        branch_3 = slim.conv2d(
            branch_3, depth(128), [1, 1],
            weights_initializer=trunc_normal(0.1),
            scope='Conv2d_0b_1x1')
      net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net

    # 14 x 14 x 576
    end_point = 'Mixed_4c'
    with tf.variable_scope(end_point):
      with tf.variable_scope('Branch_0'):
        branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
      with tf.variable_scope('Branch_1'):
        branch_1 = slim.conv2d(
            net, depth(96), [1, 1],
            weights_initializer=trunc_normal(0.09),
            scope='Conv2d_0a_1x1')
        branch_1 = slim.conv2d(branch_1, depth(128), [3, 3],
                                scope='Conv2d_0b_3x3')
      with tf.variable_scope('Branch_2'):
        branch_2 = slim.conv2d(
            net, depth(96), [1, 1],
            weights_initializer=trunc_normal(0.09),
            scope='Conv2d_0a_1x1')
        branch_2 = slim.conv2d(branch_2, depth(128), [3, 3],
                                scope='Conv2d_0b_3x3')
        branch_2 = slim.conv2d(branch_2, depth(128), [3, 3],
                                scope='Conv2d_0c_3x3')
      with tf.variable_scope('Branch_3'):
        branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
        branch_3 = slim.conv2d(
            branch_3, depth(128), [1, 1],
            weights_initializer=trunc_normal(0.1),
            scope='Conv2d_0b_1x1')
      net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net

    # 14 x 14 x 576
    end_point = 'Mixed_4d'
    with tf.variable_scope(end_point):
      with tf.variable_scope('Branch_0'):
        branch_0 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
      with tf.variable_scope('Branch_1'):
        branch_1 = slim.conv2d(
            net, depth(128), [1, 1],
            weights_initializer=trunc_normal(0.09),
            scope='Conv2d_0a_1x1')
        branch_1 = slim.conv2d(branch_1, depth(160), [3, 3],
                                scope='Conv2d_0b_3x3')
      with tf.variable_scope('Branch_2'):
        branch_2 = slim.conv2d(
            net, depth(128), [1, 1],
            weights_initializer=trunc_normal(0.09),
            scope='Conv2d_0a_1x1')
        branch_2 = slim.conv2d(branch_2, depth(160), [3, 3],
                                scope='Conv2d_0b_3x3')
        branch_2 = slim.conv2d(branch_2, depth(160), [3, 3],
                                scope='Conv2d_0c_3x3')
      with tf.variable_scope('Branch_3'):
        branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
        branch_3 = slim.conv2d(
            branch_3, depth(96), [1, 1],
            weights_initializer=trunc_normal(0.1),
            scope='Conv2d_0b_1x1')
      net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net

    # 14 x 14 x 576
    end_point = 'Mixed_4e'
    with tf.variable_scope(end_point):
      with tf.variable_scope('Branch_0'):
        branch_0 = slim.conv2d(net, depth(96), [1, 1], scope='Conv2d_0a_1x1')
      with tf.variable_scope('Branch_1'):
        branch_1 = slim.conv2d(
            net, depth(128), [1, 1],
            weights_initializer=trunc_normal(0.09),
            scope='Conv2d_0a_1x1')
        branch_1 = slim.conv2d(branch_1, depth(192), [3, 3],
                                scope='Conv2d_0b_3x3')
      with tf.variable_scope('Branch_2'):
        branch_2 = slim.conv2d(
            net, depth(160), [1, 1],
            weights_initializer=trunc_normal(0.09),
            scope='Conv2d_0a_1x1')
        branch_2 = slim.conv2d(branch_2, depth(192), [3, 3],
                                scope='Conv2d_0b_3x3')
        branch_2 = slim.conv2d(branch_2, depth(192), [3, 3],
                                scope='Conv2d_0c_3x3')
      with tf.variable_scope('Branch_3'):
        branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
        branch_3 = slim.conv2d(
            branch_3, depth(96), [1, 1],
            weights_initializer=trunc_normal(0.1),
            scope='Conv2d_0b_1x1')
      net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net

    # 14 x 14 x 576
    end_point = 'Mixed_5a'
    with tf.variable_scope(end_point):
      with tf.variable_scope('Branch_0'):
        branch_0 = slim.conv2d(
            net, depth(128), [1, 1],
            weights_initializer=trunc_normal(0.09),
            scope='Conv2d_0a_1x1')
        branch_0 = slim.conv2d(branch_0, depth(192), [3, 3], stride=2,
                                scope='Conv2d_1a_3x3')
      with tf.variable_scope('Branch_1'):
        branch_1 = slim.conv2d(
            net, depth(192), [1, 1],
            weights_initializer=trunc_normal(0.09),
            scope='Conv2d_0a_1x1')
        branch_1 = slim.conv2d(branch_1, depth(256), [3, 3],
                                scope='Conv2d_0b_3x3')
        branch_1 = slim.conv2d(branch_1, depth(256), [3, 3], stride=2,
                                scope='Conv2d_1a_3x3')
      with tf.variable_scope('Branch_2'):
        branch_2 = slim.max_pool2d(net, [3, 3], stride=2,
                                    scope='MaxPool_1a_3x3')
      net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
      end_points[end_point] = net

    # 7 x 7 x 1024
    end_point = 'Mixed_5b'
    with tf.variable_scope(end_point):
      with tf.variable_scope('Branch_0'):
        branch_0 = slim.conv2d(net, depth(352), [1, 1], scope='Conv2d_0a_1x1')
      with tf.variable_scope('Branch_1'):
        branch_1 = slim.conv2d(
            net, depth(192), [1, 1],
            weights_initializer=trunc_normal(0.09),
            scope='Conv2d_0a_1x1')
        branch_1 = slim.conv2d(branch_1, depth(320), [3, 3],
                                scope='Conv2d_0b_3x3')
      with tf.variable_scope('Branch_2'):
        branch_2 = slim.conv2d(
            net, depth(160), [1, 1],
            weights_initializer=trunc_normal(0.09),
            scope='Conv2d_0a_1x1')
        branch_2 = slim.conv2d(branch_2, depth(224), [3, 3],
                                scope='Conv2d_0b_3x3')
        branch_2 = slim.conv2d(branch_2, depth(224), [3, 3],
                                scope='Conv2d_0c_3x3')
      with tf.variable_scope('Branch_3'):
        branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
        branch_3 = slim.conv2d(
            branch_3, depth(128), [1, 1],
            weights_initializer=trunc_normal(0.1),
            scope='Conv2d_0b_1x1')
      net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net

    # 7 x 7 x 1024
    end_point = 'Mixed_5c'
    with tf.variable_scope(end_point):
      with tf.variable_scope('Branch_0'):
        branch_0 = slim.conv2d(net, depth(352), [1, 1], scope='Conv2d_0a_1x1')
      with tf.variable_scope('Branch_1'):
        branch_1 = slim.conv2d(
            net, depth(192), [1, 1],
            weights_initializer=trunc_normal(0.09),
            scope='Conv2d_0a_1x1')
        branch_1 = slim.conv2d(branch_1, depth(320), [3, 3],
                                scope='Conv2d_0b_3x3')
      with tf.variable_scope('Branch_2'):
        branch_2 = slim.conv2d(
            net, depth(192), [1, 1],
            weights_initializer=trunc_normal(0.09),
            scope='Conv2d_0a_1x1')
        branch_2 = slim.conv2d(branch_2, depth(224), [3, 3],
                                scope='Conv2d_0b_3x3')
        branch_2 = slim.conv2d(branch_2, depth(224), [3, 3],
                                scope='Conv2d_0c_3x3')
      with tf.variable_scope('Branch_3'):
        branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
        branch_3 = slim.conv2d(
            branch_3, depth(128), [1, 1],
            weights_initializer=trunc_normal(0.1),
            scope='Conv2d_0b_1x1')
      net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net
    
    # 7 x 7 x 1024
    end_point = 'Logits'
    with tf.variable_scope(end_point):
      net = slim.avg_pool2d(net, [7, 7], padding='VALID', scope='AvgPool_1a_7x7')
      # 1 x 1 x 1024
      net = slim.dropout(net, keep_prob=DROPOUT_KEEP, scope='Dropout_1b')
      net = slim.conv2d(net, NUM_CLASSES, [1, 1], activation_fn=None,
                             normalizer_fn=None, scope='Conv2d_1c_1x1')
      net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
      end_points[end_point] = net
  return net, end_points


@ops.RegisterGradient("FirstGradOnly_GradFn")
def firstGradOnly_GradFn(op, grad):
    return [grad, None]

def stochastic_branch_fn(input_tensor, slope_tensor, is_training):
  batch_size, branch_num = input_tensor.shape.as_list()
  if is_training:
    probs_tensor = tf.nn.softmax(input_tensor * slope_tensor, dim=1, name='branch_prob')
    with tf.get_default_graph().gradient_override_map({'Cumsum': 'FirstGradOnly_GradFn', 'HardGate': 'Identity', 'Mul': 'FirstGradOnly_GradFn'}):
      low_limit = tf.cumsum(probs_tensor, axis=1, exclusive=True, name='low_limit')
      high_limit = tf.cumsum(probs_tensor, axis=1, exclusive=False, name='high_limit')
      rand_number = tf.stop_gradient(tf.random_uniform((batch_size, 1)))
      low_bound = tf.stop_gradient(1.0 - tf.hard_gate(low_limit - rand_number))
      high_bound = tf.hard_gate(high_limit - rand_number)
      result = tf.multiply(high_bound, low_bound, name='stochastic_branch')
    return result
  else:
    choice_tensor = tf.argmax(input_tensor, axis=1, name='most_prob_branch_indx')
    one_hot_result = tf.one_hot(choice_tensor, branch_num, name='most_prob_branch')
    return one_hot_result


def inception_v2_gate_seg(inputs, end_points, branch_num):
  depth = lambda d: max(int(d), 16)
  with slim.arg_scope(
      [slim.conv2d, slim.max_pool2d, slim.avg_pool2d, slim.separable_conv2d],
      stride=1, padding='SAME'):

    # 28 x 28 x 192
    end_point = 'Mixed_4a'
    with tf.variable_scope(end_point):
      with tf.variable_scope('Branch_0'):
        branch_0 = slim.conv2d(
            inputs, depth(128), [1, 1],
            weights_initializer=trunc_normal(0.09),
            scope='Conv2d_0a_1x1')
        branch_0 = slim.conv2d(branch_0, depth(160), [3, 3], stride=2,
                                scope='Conv2d_1a_3x3')
      with tf.variable_scope('Branch_1'):
        branch_1 = slim.conv2d(
            inputs, depth(64), [1, 1],
            weights_initializer=trunc_normal(0.09),
            scope='Conv2d_0a_1x1')
        branch_1 = slim.conv2d(
            branch_1, depth(96), [3, 3], scope='Conv2d_0b_3x3')
        branch_1 = slim.conv2d(
            branch_1, depth(96), [3, 3], stride=2, scope='Conv2d_1a_3x3')
      with tf.variable_scope('Branch_2'):
        branch_2 = slim.max_pool2d(
            inputs, [3, 3], stride=2, scope='MaxPool_1a_3x3')
      net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
      end_points[end_point] = net

    # 14 x 14 x 448
    end_point = 'Mixed_5a'
    with tf.variable_scope(end_point):
      with tf.variable_scope('Branch_0'):
        branch_0 = slim.conv2d(
            net, depth(128), [1, 1],
            weights_initializer=trunc_normal(0.09),
            scope='Conv2d_0a_1x1')
        branch_0 = slim.conv2d(branch_0, depth(192), [3, 3], stride=2,
                                scope='Conv2d_1a_3x3')
      with tf.variable_scope('Branch_1'):
        branch_1 = slim.conv2d(
            net, depth(192), [1, 1],
            weights_initializer=trunc_normal(0.09),
            scope='Conv2d_0a_1x1')
        branch_1 = slim.conv2d(branch_1, depth(256), [3, 3],
                                scope='Conv2d_0b_3x3')
        branch_1 = slim.conv2d(branch_1, depth(256), [3, 3], stride=2,
                                scope='Conv2d_1a_3x3')
      with tf.variable_scope('Branch_2'):
        branch_2 = slim.max_pool2d(net, [3, 3], stride=2,
                                    scope='MaxPool_1a_3x3')
      net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
      end_points[end_point] = net
    
    # 7 x 7 x 896
    end_point = 'Logits'
    with tf.variable_scope(end_point):
      net = slim.avg_pool2d(net, [7, 7], padding='VALID', scope='AvgPool_1a_7x7')
      # 1 x 1 x 1024
      # net = slim.dropout(net, keep_prob=DROPOUT_KEEP, scope='Dropout_1b')
      net = slim.conv2d(net, branch_num, [1, 1], activation_fn=None,
                             normalizer_fn=None, scope='Conv2d_1c_1x1')
      net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
      end_points[end_point] = net
  return net, end_points


def inception_v2(inputs,
                 slope_tensor,
                 num_branches,
                 is_training=True,
                 reuse=True,
                 scope='InceptionV2'):
  """Inception v2 model for classification.
  Constructs an Inception v2 network for classification as described in
  http://arxiv.org/abs/1502.03167.
  The default image size used to train this network is 224x224.
  Args:
    inputs: a tensor of shape [batch_size, height, width, channels].
    is_training: whether is training or not.
    dropout_keep_prob: the percentage of activation values that are retained.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.
  Returns:
    logits: the pre-softmax activations, a tensor of size
      [batch_size, num_classes]
    end_points: a dictionary from components of the network to the corresponding
      activation.
  Raises:
    ValueError: if final_endpoint is not set to one of the predefined values,
                or depth_multiplier <= 0
  """

  # Final pooling and prediction
  with tf.variable_scope(scope, 'InceptionV2', [inputs], reuse=reuse) as scope:
    with slim.arg_scope(inception_arg_scope()):
      end_points = {}
      net = inputs
      with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=(is_training and TRAIN_COMMON_SEG)):
        tmp_end_points = {}
        net, tmp_end_points = inception_v2_common_seg(net, tmp_end_points)
        end_points['common_seg'] = tmp_end_points
      with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
        with tf.variable_scope('CondBranchFn', values=[net]):
          tmp_end_points = {}
          branch_val, tmp_end_points = inception_v2_gate_seg(net, tmp_end_points, num_branches)
          end_points['branch_fn'] = tmp_end_points
          end_points['branch_preact'] = branch_val
          branch_val = stochastic_branch_fn(branch_val, slope_tensor, is_training)
          end_points['branch_result'] = branch_val
          branch_decision_list = tf.unstack(branch_val, axis=1, name='branch_decisions')
        branch_endpoints = []
        for i in xrange(num_branches):
          branch_name = 'CondBranch_%i' % i
          with tf.variable_scope(branch_name, values=[net]):
            with slim.arg_scope([slim.batch_norm], batch_weights=branch_decision_list[i]):
              tmp_end_points = {}
              branch_output, tmp_end_points = inception_v2_branch_seg(net, tmp_end_points)
              end_points[branch_name] = tmp_end_points
              branch_endpoints.append(branch_output)
        all_branch_output = tf.stack(branch_endpoints, axis=1, name='all_branch_output')
        end_points['all_branch_output'] = all_branch_output
        final_output = tf.reduce_sum(tf.multiply(all_branch_output, tf.expand_dims(branch_val, axis=2)), axis=1, name='final_output')
        end_points['final_output'] = final_output
        end_points['soft_prediction'] = tf.reduce_sum(tf.multiply(tf.nn.softmax(all_branch_output, dim=2), 
            tf.expand_dims(tf.nn.softmax(end_points['branch_preact'], dim=1), axis=2)), axis=1, name='soft_prediction')
        end_points['hard_prediction'] = tf.argmax(final_output, axis=1, name='hard_prediction')
        return final_output, end_points

inception_v2.default_image_size = 224
