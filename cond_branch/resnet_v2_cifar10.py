
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf
from tensorflow.python.framework import ops

slim = tf.contrib.slim

NUM_CLASSES=10

WEIGHT_DECAY=0.0001
BATCH_NORM_DECAY=0.99
BATCH_NORM_EPSILON=1e-3

TRAIN_COMMON_SEG=True

BATCH_NORM_PARAMS = {
      # Decay for the moving averages.
      'decay': BATCH_NORM_DECAY,
      # epsilon to prevent 0s in variance.
      'epsilon': BATCH_NORM_EPSILON,
      # collection containing update_ops.
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
}


def conv2d_same(inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
  """Strided 2-D convolution with 'SAME' padding.

  When stride > 1, then we do explicit zero-padding, followed by conv2d with
  'VALID' padding.

  Note that

     net = conv2d_same(inputs, num_outputs, 3, stride=stride)

  is equivalent to

     net = slim.conv2d(inputs, num_outputs, 3, stride=1, padding='SAME')
     net = subsample(net, factor=stride)

  whereas

     net = slim.conv2d(inputs, num_outputs, 3, stride=stride, padding='SAME')

  is different when the input's height or width is even, which is why we add the
  current function. For more details, see ResnetUtilsTest.testConv2DSameEven().

  Args:
    inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
    num_outputs: An integer, the number of output filters.
    kernel_size: An int with the kernel_size of the filters.
    stride: An integer, the output stride.
    rate: An integer, rate for atrous convolution.
    scope: Scope.

  Returns:
    output: A 4-D tensor of size [batch, height_out, width_out, channels] with
      the convolution output.
  """
  if stride == 1:
    return slim.conv2d(inputs, num_outputs, kernel_size, stride=1, rate=rate,
                       padding='SAME', scope=scope)
  else:
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    inputs = tf.pad(inputs,
                    [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                       rate=rate, padding='VALID', scope=scope)


def resnet_arg_scope():
  """Defines the default ResNet arg scope.

  TODO(gpapan): The batch-normalization related default values above are
    appropriate for use in conjunction with the reference ResNet models
    released at https://github.com/KaimingHe/deep-residual-networks. When
    training ResNets from scratch, they might need to be tuned.

  Returns:
    An `arg_scope` to use for the resnet models.
  """
  with slim.arg_scope(
      [slim.conv2d],
      weights_regularizer=slim.l2_regularizer(WEIGHT_DECAY),
      weights_initializer=slim.variance_scaling_initializer(),
      activation_fn=tf.nn.relu,
      normalizer_fn=slim.batch_norm,
      normalizer_params=BATCH_NORM_PARAMS) as arg_sc:
    return arg_sc


def bottleneck(inputs, increase_dim, double_depth, scope, output_collection):
  with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    # preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
    if double_depth:
      depth_out = depth_in * 2
    else:
      depth_out = depth_in
    if increase_dim:
      stride = 2
    else:
      stride = 1
    
    if increase_dim:
      shortcut = tf.nn.avg_pool(inputs, [1, 1, 1, 1], [1, 2, 2, 1], 'SAME')
    else:
      shortcut = inputs
    if double_depth:
      shortcut = tf.pad(shortcut, [[0, 0], [0, 0], [0, 0], [0, depth_in]])
    
    residual = conv2d_same(inputs, depth_out, 3, stride, scope='conv1')
    residual = slim.conv2d(residual, depth_out, [3, 3], stride=1,
                           activation_fn=None, padding='SAME', scope='conv2')

    output = shortcut + residual
    output_collection[sc.name] = output
  return output


def resnet_v2_cifar10_root_block(inputs, end_points):
  net = slim.conv2d(inputs, 16, [3, 3], stride=1, padding='SAME', scope='root_block')
  end_points['root_block'] = net
  return net

def resnet_v2_cifar10_ending_block(inputs, end_points, num_classes):
  net = tf.reduce_mean(inputs, [1, 2], name='GlobalPool', keep_dims=True)
  net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='logits')
  net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
  end_points['ending_block'] = net
  return net

def resnet_v2_cifar10_stack_blocks(inputs, end_points, schema, exempt_first=False, double_depth=True):
  is_first = True
  net = inputs
  for c, i in enumerate(schema):
    with tf.variable_scope('stage_%i' % c, None, [net]):
      if is_first and exempt_first:
        is_first = False
        net = bottleneck(net, False, False, 'block_0', end_points)
      else:
        net = bottleneck(net, True, double_depth, 'block_0', end_points)
      for j in xrange(1, i):
        net = bottleneck(net, False, False, 'block_%i' % j, end_points)
  return net


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


def resnet_v2_cifar(inputs,
                 slope_tensor,
                 common_seg_scheme,
                 branch_seg_scheme,
                 branch_double_depth,
                 num_branches,
                 is_training=True,
                 reuse=False,
                 scope='ResNetV2'):

  # Final pooling and prediction
  with tf.variable_scope(scope, 'ResNetV2', [inputs], reuse=reuse) as scope:
    with slim.arg_scope(resnet_arg_scope()):
      end_points = {}
      net = inputs
      with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=(is_training and TRAIN_COMMON_SEG)):
        tmp_end_points = {}
        net = resnet_v2_cifar10_root_block(net, tmp_end_points)
        net = resnet_v2_cifar10_stack_blocks(net, tmp_end_points, common_seg_scheme, exempt_first=True)
        end_points['common_seg'] = tmp_end_points
      with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
        with tf.variable_scope('CondBranchFn', values=[net]):
          tmp_end_points = {}
          branch_val = resnet_v2_cifar10_stack_blocks(net, tmp_end_points, [1, 1])
          branch_val = resnet_v2_cifar10_ending_block(branch_val, tmp_end_points, num_branches)
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
              branch_output = resnet_v2_cifar10_stack_blocks(net, tmp_end_points, branch_seg_scheme, double_depth=branch_double_depth)
              branch_output = resnet_v2_cifar10_ending_block(branch_output, tmp_end_points, NUM_CLASSES)
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


# def resnet_v2_cifar_no_branch(inputs,
#                  slope_tensor,
#                  num_branches,
#                  is_training=True,
#                  reuse=False,
#                  scope='ResNetV2'):

#   # Final pooling and prediction
#   with tf.variable_scope(scope, 'ResNetV2', [inputs], reuse=reuse) as scope:
#     with slim.arg_scope(resnet_arg_scope()):
#       end_points = {}
#       net = inputs
#       with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=(is_training)):
#         net = resnet_v2_cifar10_root_block(net, end_points)
#         net = resnet_v2_cifar10_stack_blocks(net, end_points, [NET_SIZE_N, NET_SIZE_N, NET_SIZE_N], exempt_first=True)
#         net = resnet_v2_cifar10_ending_block(net, end_points, NUM_CLASSES)
#       final_output = net
#       end_points['final_output'] = final_output
#       end_points['hard_prediction'] = tf.argmax(final_output, axis=1, name='hard_prediction')
#       end_points['soft_prediction'] = tf.nn.softmax(final_output)
#       return final_output, end_points
