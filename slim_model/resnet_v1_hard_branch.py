# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for the original form of Residual Networks.

The 'v1' residual networks (ResNets) implemented in this module were proposed
by:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

Other variants were introduced in:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The networks defined in this module utilize the bottleneck building block of
[1] with projection shortcuts only for increasing depths. They employ batch
normalization *after* every weight layer. This is the architecture used by
MSRA in the Imagenet and MSCOCO 2016 competition models ResNet-101 and
ResNet-152. See [2; Fig. 1a] for a comparison between the current 'v1'
architecture and the alternative 'v2' architecture of [2] which uses batch
normalization *before* every weight layer in the so-called full pre-activation
units.

Typical use:

   from tensorflow.contrib.slim.nets import resnet_v1

ResNet-101 for image classification into 1000 classes:

   # inputs has shape [batch, 224, 224, 3]
   with slim.arg_scope(resnet_v1.resnet_arg_scope()):
      net, end_points = resnet_v1.resnet_v1_101(inputs, 1000, is_training=False)

ResNet-101 for semantic segmentation into 21 classes:

   # inputs has shape [batch, 513, 513, 3]
   with slim.arg_scope(resnet_v1.resnet_arg_scope()):
      net, end_points = resnet_v1.resnet_v1_101(inputs,
                                                21,
                                                is_training=False,
                                                global_pool=False,
                                                output_stride=16)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf
from tensorflow.python.framework import ops

import resnet_utils


resnet_arg_scope = resnet_utils.resnet_arg_scope
slim = tf.contrib.slim


@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, rate=1,
               outputs_collections=None, scope=None):
  """Bottleneck residual unit variant with BN after convolutions.

  This is the original residual unit proposed in [1]. See Fig. 1(a) of [2] for
  its definition. Note that we use here the bottleneck variant which has an
  extra bottleneck layer.

  When putting together two consecutive ResNet blocks that use this unit, one
  should use stride = 2 in the last unit of the first block.

  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth: The depth of the ResNet unit output.
    depth_bottleneck: The depth of the bottleneck layers.
    stride: The ResNet unit's stride. Determines the amount of downsampling of
      the units output compared to its input.
    rate: An integer, rate for atrous convolution.
    outputs_collections: Collection to add the ResNet unit output.
    scope: Optional variable_scope.

  Returns:
    The ResNet unit's output.
  """
  with tf.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    if depth == depth_in:
      shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
    else:
      shortcut = slim.conv2d(inputs, depth, [1, 1], stride=stride,
                             activation_fn=None, scope='shortcut')

    residual = slim.conv2d(inputs, depth_bottleneck, [1, 1], stride=1,
                           scope='conv1')
    residual = resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride,
                                        rate=rate, scope='conv2')
    residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                           activation_fn=None, scope='conv3')

    output = tf.nn.relu(shortcut + residual)

    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.original_name_scope,
                                            output)

batch_norm_params = {
      'decay': 0.997,
      'epsilon': 1e-5,
      'scale': True,
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
}


def resnet_v1_hard_branch(inputs,
              blocks1, blocks_branch, blocks2,
              num_classes=None,
              is_training=True,
              reuse=None,
              scope=None):
  """Generator for v1 ResNet models.

  This function generates a family of ResNet v1 models. See the resnet_v1_*()
  methods for specific model instantiations, obtained by selecting different
  block instantiations that produce ResNets of various depths.

  Training for image classification on Imagenet is usually done with [224, 224]
  inputs, resulting in [7, 7] feature maps at the output of the last ResNet
  block for the ResNets defined in [1] that have nominal stride equal to 32.
  However, for dense prediction tasks we advise that one uses inputs with
  spatial dimensions that are multiples of 32 plus 1, e.g., [321, 321]. In
  this case the feature maps at the ResNet output will have spatial shape
  [(height - 1) / output_stride + 1, (width - 1) / output_stride + 1]
  and corners exactly aligned with the input image corners, which greatly
  facilitates alignment of the features to the image. Using as input [225, 225]
  images results in [8, 8] feature maps at the output of the last ResNet block.

  For dense prediction tasks, the ResNet needs to run in fully-convolutional
  (FCN) mode and global_pool needs to be set to False. The ResNets in [1, 2] all
  have nominal stride equal to 32 and a good choice in FCN mode is to use
  output_stride=16 in order to increase the density of the computed features at
  small computational and memory overhead, cf. http://arxiv.org/abs/1606.00915.

  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    blocks: A list of length equal to the number of ResNet blocks. Each element
      is a resnet_utils.Block object describing the units in the block.
    num_classes: Number of predicted classes for classification tasks. If None
      we return the features before the logit layer.
    is_training: whether is training or not.
    global_pool: If True, we perform global average pooling before computing the
      logits. Set to True for image classification, False for dense prediction.
    output_stride: If None, then the output will be computed at the nominal
      network stride. If output_stride is not None, it specifies the requested
      ratio of input to output spatial resolution.
    include_root_block: If True, include the initial convolution followed by
      max-pooling, if False excludes it.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.

  Returns:
    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
      If global_pool is False, then height_out and width_out are reduced by a
      factor of output_stride compared to the respective height_in and width_in,
      else both height_out and width_out equal one. If num_classes is None, then
      net is the output of the last ResNet block, potentially after global
      average pooling. If num_classes is not None, net contains the pre-softmax
      activations.
    end_points: A dictionary from components of the network to the corresponding
      activation.

  Raises:
    ValueError: If the target output_stride is not valid.
  """
  with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.name + '_end_points'
    with slim.arg_scope([slim.conv2d, bottleneck,
                         resnet_utils.stack_blocks_dense],
                        outputs_collections=end_points_collection):
      with slim.arg_scope([slim.batch_norm], is_training=False):
        net = inputs
        net = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
        net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
        net = resnet_utils.stack_blocks_dense(net, blocks1)
      with slim.arg_scope([slim.batch_norm], is_training=is_training):
        with tf.variable_scope('branch_fn', [net], reuse=reuse):
          # branch = tf.stop_gradient(net)
          branch = net
          branch = resnet_utils.stack_blocks_dense(branch, blocks_branch)
          branch = tf.reduce_mean(branch, [1, 2], name='pool_branch', keep_dims=True)
          branch_prob = slim.conv2d(branch, 1, [1, 1], activation_fn=tf.sigmoid, scope='branch_preact')
          if is_training:
            branch = branch_prob - tf.stop_gradient(tf.random_uniform(tf.shape(branch_prob)))
          else:
            branch = branch_prob - 0.5
          branch = tf.hard_gate(branch, name='branch_val')
          branch_size = branch.shape.dims[0].value
          branch = tf.constant([0.0] * (branch_size // 2) + [1.0] * (branch_size - branch_size//2),
                               tf.float32, shape=branch.shape)
        with tf.variable_scope('left_branch', [net], reuse=reuse):
          with slim.arg_scope([slim.batch_norm], batch_weights=(1.0-tf.squeeze(branch))):
            net_left = resnet_utils.stack_blocks_dense(net, blocks2)
            net_left = tf.reduce_mean(net_left, [1, 2], name='pool5', keep_dims=True)
            net_left = slim.conv2d(net_left, num_classes, [1, 1], activation_fn=None,
                              normalizer_fn=None, scope='logits')
        with tf.variable_scope('right_branch', [net], reuse=reuse):
          with slim.arg_scope([slim.batch_norm], batch_weights=(tf.squeeze(branch))):
            net_right = resnet_utils.stack_blocks_dense(net, blocks2)
            net_right = tf.reduce_mean(net_right, [1, 2], name='pool5', keep_dims=True)
            net_right = slim.conv2d(net_right, num_classes, [1, 1], activation_fn=None,
                              normalizer_fn=None, scope='logits')

        net = net_left * (1.0 - branch) + net_right * branch

        # Convert end_points_collection into a dictionary of end_points.
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        end_points['predictions'] = slim.softmax(net, scope='predictions')
        end_points['branch'] = branch
        end_points['branch_prob'] = branch_prob
        return net, end_points


def v1(inputs,
                  num_classes=None,
                  is_training=True,
                  reuse=None,
                  scope='resnet_v1_50'):
  blocks1 = [
      resnet_utils.Block(
          'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
      resnet_utils.Block(
          'block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
  ]
  blocks_branch = [
    resnet_utils.Block(
          'block_br1', bottleneck, [(1024, 256, 2)]),
      resnet_utils.Block(
          'block_br2', bottleneck, [(2048, 512, 1)])
  ]
  blocks2 = [
      resnet_utils.Block(
          'block3', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
      resnet_utils.Block(
          'block4', bottleneck, [(2048, 512, 1)] * 3)
  ]
  return resnet_v1_hard_branch(inputs, blocks1, blocks_branch, blocks2,
                   num_classes=num_classes, is_training=is_training,
                   reuse=reuse, scope=scope)

def v1_fn(num_classes, weight_decay=0.0, is_training=False):
  """Returns a network_fn such as `logits, end_points = network_fn(images)`.

  Args:
    name: The name of the network.
    num_classes: The number of classes to use for classification.
    weight_decay: The l2 coefficient for the model weights.
    is_training: `True` if the model is being used for training and `False`
      otherwise.

  Returns:
    network_fn: A function that applies the model to a batch of images. It has
      the following signature:
        logits, end_points = network_fn(images)
  Raises:
    ValueError: If network `name` is not recognized.
  """

  arg_scope = resnet_utils.resnet_arg_scope(weight_decay=weight_decay)
  func = v1
  @functools.wraps(func)
  def network_fn(images):
    with slim.arg_scope(arg_scope):
      return func(images, num_classes, is_training=is_training)

  return network_fn