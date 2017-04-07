

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import tensorflow as tf

from tensorflow.python.ops import control_flow_ops
import imagenet
import resnet_v1_hard_branch
import vgg_preprocessing

slim = tf.contrib.slim

###############################################################################
# Need to Specify
###############################################################################
tf.app.flags.DEFINE_string(
    'train_dir', None,
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 8,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 300,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 300,
    'The frequency with which the model is saved, in seconds.')

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
    'weight_decay', 0.0, 'The weight decay on the model weights.') # 0.00004

tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.8, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 1.0,
    'Number of epochs after which learning rate decays.')

#######################
# Dataset Flags #
#######################

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 1,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_integer(
    'batch_size', 64, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'train_image_size', 224, 'Train image size')

#####################
# Fine-Tuning Flags #
#####################

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')

FLAGS = tf.app.flags.FLAGS


def _configure_learning_rate(num_samples_per_epoch, global_step):
  """Configures the learning rate.

  Args:
    num_samples_per_epoch: The number of samples in each epoch of training.
    global_step: The global_step tensor.

  Returns:
    A `Tensor` representing the learning rate.

  Raises:
    ValueError: if
  """
  decay_steps = int(num_samples_per_epoch / FLAGS.batch_size *
                    FLAGS.num_epochs_per_decay)

  if FLAGS.learning_rate_decay_type == 'exponential':
    return tf.train.exponential_decay(FLAGS.learning_rate,
                                      global_step,
                                      decay_steps,
                                      FLAGS.learning_rate_decay_factor,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'fixed':
    return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'polynomial':
    return tf.train.polynomial_decay(FLAGS.learning_rate,
                                     global_step,
                                     decay_steps,
                                     FLAGS.end_learning_rate,
                                     power=1.0,
                                     cycle=False,
                                     name='polynomial_decay_learning_rate')
  else:
    raise ValueError('learning_rate_decay_type [%s] was not recognized',
                     FLAGS.learning_rate_decay_type)


def _configure_optimizer(learning_rate):
  """Configures the optimizer used for training.

  Args:
    learning_rate: A scalar or `Tensor` learning rate.

  Returns:
    An instance of an optimizer.

  Raises:
    ValueError: if FLAGS.optimizer is not recognized.
  """
  if FLAGS.optimizer == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(
        learning_rate,
        rho=FLAGS.adadelta_rho,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
  elif FLAGS.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=FLAGS.adam_beta1,
        beta2=FLAGS.adam_beta2,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'ftrl':
    optimizer = tf.train.FtrlOptimizer(
        learning_rate,
        learning_rate_power=FLAGS.ftrl_learning_rate_power,
        initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
        l1_regularization_strength=FLAGS.ftrl_l1,
        l2_regularization_strength=FLAGS.ftrl_l2)
  elif FLAGS.optimizer == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate,
        momentum=FLAGS.momentum,
        name='Momentum')
  elif FLAGS.optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=FLAGS.rmsprop_decay,
        momentum=FLAGS.rmsprop_momentum,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  else:
    raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
  return optimizer


def _get_init_fn():
  """Returns a function run by the chief worker to warm-start the training.

  Note that the init_fn is only run when initializing the model during the very
  first global step.

  Returns:
    An init function run by the supervisor.
  """
  if FLAGS.checkpoint_path is None:
    return None

  # Warn the user if a checkpoint exists in the train_dir. Then we'll be
  # ignoring the checkpoint anyway.
  if tf.train.latest_checkpoint(FLAGS.train_dir):
    tf.logging.info(
        'Ignoring --checkpoint_path because a checkpoint already exists in %s'
        % FLAGS.train_dir)
    return None

  # TODO(sguada) variables.filter_variables()
  variables_to_restore1 = {}
  variables_to_restore2 = {}
  for var in slim.get_model_variables():
    var_name = var.op.name
    if ('/branch_fn' not in var_name) and ('/right_branch' not in var_name):
        var_name = var_name.replace('/left_branch', '')
        variables_to_restore1[var_name] = var
    elif '/right_branch' in var_name:
        var_name = var_name.replace('/right_branch', '')
        variables_to_restore2[var_name] = var

  if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
    checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
  else:
    checkpoint_path = FLAGS.checkpoint_path

  tf.logging.info('Fine-tuning from %s' % checkpoint_path)

  init_f1 = slim.assign_from_checkpoint_fn(
                checkpoint_path,
                variables_to_restore1)
  init_f2 = slim.assign_from_checkpoint_fn(
                checkpoint_path,
                variables_to_restore2)

  def _init_fn(sess):
      init_f1(sess)
      init_f2(sess)

  return _init_fn


def _get_variables_to_train():
  """Returns a list of variables to train.

  Returns:
    A list of variables to train by the optimizer.
  """
#   return tf.trainable_variables()
  return [v for v in tf.trainable_variables() if v.name.startswith('resnet_v1_50/left_branch') or
                                                 v.name.startswith('resnet_v1_50/right_branch') or
                                                 v.name.startswith('resnet_v1_50/branch_fn')]

def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')
  if not FLAGS.train_dir:
    raise ValueError('You must supply the logging directory with --train_dir')
#   if not FLAGS.checkpoint_path:
#     raise ValueError('You must supply the pretrained model with --checkpoint_path')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    # Create global_step
    global_step = slim.create_global_step()

    dataset = imagenet.get_split(FLAGS.dataset_split_name, FLAGS.dataset_dir)

    ####################
    # Select the network #
    ####################
    network_fn = resnet_v1_hard_branch.v1_fn(
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        weight_decay=FLAGS.weight_decay,
        is_training=True)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers=FLAGS.num_readers,
        common_queue_capacity=20 * FLAGS.batch_size,
        common_queue_min=10 * FLAGS.batch_size)
    [image, label] = provider.get(['image', 'label'])
    label -= FLAGS.labels_offset

    train_image_size = FLAGS.train_image_size

    image = vgg_preprocessing.preprocess_image(image, train_image_size, train_image_size, is_training=True)

    images, labels = tf.train.batch(
        [image, label],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=5 * FLAGS.batch_size)
    labels = slim.one_hot_encoding(
        labels, dataset.num_classes - FLAGS.labels_offset)
    batch_queue = slim.prefetch_queue.prefetch_queue(
        [images, labels], capacity=10)


    ####################
    # Define the model #
    ####################
    def net_fn(batch_queue):
      """Allows data parallelism by creating multiple clones of network_fn."""
      images, labels = batch_queue.dequeue()
      logits, end_points = network_fn(images)

      #############################
      # Specify the loss function #
      #############################

      slim.losses.softmax_cross_entropy(
          tf.squeeze(logits), labels, label_smoothing=FLAGS.label_smoothing, weights=1.0)
    #   current_bias = tf.abs(tf.reduce_mean(end_points['branch_prob']) - 0.5)
    #   tf.losses.compute_weighted_loss(tf.nn.relu(current_bias - 0.1), weights=10.0)
      predictions = tf.argmax(tf.squeeze(logits), 1)
      accuracy = tf.reduce_mean(tf.to_float(tf.equal(predictions, tf.argmax(labels, axis=1))))
      weight_one = tf.constant(1.0, tf.float32)
      moving_accuracy = tf.train.ExponentialMovingAverage(decay=0.99)
      moving_accuracy_op = moving_accuracy.apply([accuracy, weight_one])
      tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, moving_accuracy_op)
      with tf.control_dependencies(moving_accuracy_op):
        end_points['moving_accuracy'] = moving_accuracy.average(accuracy) / moving_accuracy.average(weight_one)
      return end_points

    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    net = net_fn(batch_queue)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    # Add summaries for end_points.
    for end_point in net:
      x = net[end_point]
      summaries.add(tf.summary.histogram('activations/' + end_point, x))
    #   summaries.add(tf.summary.scalar('sparsity/' + end_point,
    #                                   tf.nn.zero_fraction(x)))

    # Add summaries for losses.
    for loss in tf.get_collection(tf.GraphKeys.LOSSES):
      summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

    # Add summaries for variables.
    for variable in slim.get_model_variables():
      summaries.add(tf.summary.histogram(variable.op.name, variable))

    #########################################
    # Configure the optimization procedure. #
    #########################################
    learning_rate = _configure_learning_rate(dataset.num_samples, global_step)
    optimizer = _configure_optimizer(learning_rate)
    summaries.add(tf.summary.scalar('learning_rate', learning_rate))

    # Variables to train.
    variables_to_train = _get_variables_to_train()

    #  and returns a train_tensor and summary_op
    pred_loss = tf.add_n(tf.get_collection(tf.GraphKeys.LOSSES), name='Pred_Loss')
    if len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) > 0:
        reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='Reg_Loss')
    else:
        reg_loss = 0.0
    total_loss = tf.add(pred_loss, reg_loss, name='total_loss')
    gradients = optimizer.compute_gradients(total_loss, var_list=variables_to_train)
    # Add total_loss to summary.
    summaries.add(tf.summary.scalar('total_loss', total_loss))

    # Create gradient updates.
    grad_updates = optimizer.apply_gradients(gradients,
                                             global_step=global_step)
    update_ops.append(grad_updates)

    update_op = tf.group(*update_ops)

    total_loss = tf.Print(total_loss, [net['moving_accuracy']], 'Moving_Accuracy')

    train_tensor = control_flow_ops.with_dependencies([update_op], total_loss,
                                                      name='train_op')

    # Add the summaries from the first clone. These contain the summaries
    # created by model_fn and either optimize_clones() or _gather_clone_loss().
    summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    # Merge all summaries together.
    summary_op = tf.summary.merge(list(summaries), name='summary_op')


    ###########################
    # Kicks off the training. #
    ###########################
    slim.learning.train(
        train_tensor,
        logdir=FLAGS.train_dir,
        init_fn=_get_init_fn(),
        summary_op=summary_op,
        log_every_n_steps=FLAGS.log_every_n_steps,
        save_summaries_secs=FLAGS.save_summaries_secs,
        save_interval_secs=FLAGS.save_interval_secs)


if __name__ == '__main__':
  tf.app.run()
