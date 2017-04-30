from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math

from tensorflow.python.ops import control_flow_ops
import cifar10
import preprocessing_cifar10
import resnet_v2_cifar10

slim = tf.contrib.slim

NUM_BRANCHES=2
NET_SIZE_N=1

BATCH_SIZE=128

INITIAL_LEARNING_RATE=1e-1
DECAY_RATE=0.1
DECAY_STEP=160
INFLAT_RATE=1.0
INFLAT_STEP=10
MOMENTUM_RATE=0.9

TOTAL_EPOCHS=1

BATCH_SIZE_TEST=500

DISK_READER=2
PREPROCESSOR=2
TRAIN_IMAGE_SIZE=32

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_string(
    'model_log_dir', None,
    'Directory where checkpoints and event logs are written to.')

FLAGS = tf.app.flags.FLAGS


def prepare_dataset():
    with tf.device('/cpu:0'):
        with tf.variable_scope('training_data_provider'):
            dataset = cifar10.get_split('train', FLAGS.dataset_dir)

            provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset,
                num_readers=DISK_READER,
                common_queue_capacity=20 * BATCH_SIZE,
                common_queue_min=10 * BATCH_SIZE)
            [image, label] = provider.get(['image', 'label'])

            image = preprocessing_cifar10.preprocess_image(image, TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE, is_training=True)

            images, labels = tf.train.batch(
                [image, label],
                batch_size=BATCH_SIZE,
                num_threads=PREPROCESSOR,
                capacity=10 * BATCH_SIZE)
            labels = slim.one_hot_encoding(
                labels, dataset.num_classes)
            batch_queue = slim.prefetch_queue.prefetch_queue(
                [images, labels], capacity=10)
    return dataset, batch_queue


def prepare_dataset_eval():
    with tf.device('/cpu:0'):
        with tf.variable_scope('evaluation_data_provider'):

            dataset = cifar10.get_split('test', FLAGS.dataset_dir)

            provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset, shuffle=False,
                num_readers=DISK_READER,
                common_queue_capacity=20 * BATCH_SIZE_TEST,
                common_queue_min=10 * BATCH_SIZE_TEST)
            [image, label] = provider.get(['image', 'label'])

            image = preprocessing_cifar10.preprocess_image(image, TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE, is_training=False)

            images, labels = tf.train.batch(
                [image, label],
                batch_size=BATCH_SIZE_TEST,
                num_threads=PREPROCESSOR,
                capacity=10 * BATCH_SIZE_TEST)

    return dataset, images, labels


def prepare_net(batch_queue, num_samples):
    global_step = slim.create_global_step()

    batch_num = num_samples // BATCH_SIZE
    learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, DECAY_STEP * batch_num, DECAY_RATE, staircase=True, name='learning_rate')
    slope_rate = tf.train.exponential_decay(1.0, global_step, INFLAT_STEP * batch_num, INFLAT_RATE, staircase=True, name='slope_rate')

    images, labels = batch_queue.dequeue()

    logits, end_points = resnet_v2_cifar10.resnet_v2_cifar(images, slope_rate, NET_SIZE_N, NUM_BRANCHES, is_training=True)

    with tf.variable_scope('stats'):
        pred_loss = tf.losses.softmax_cross_entropy(labels, logits, scope='cross_entropy_loss')
        hard_accuracy = tf.reduce_mean(tf.to_float(tf.equal(end_points['hard_prediction'], tf.argmax(labels, axis=1))), name='hard_acc')
        soft_accuracy = tf.reduce_mean(tf.reduce_sum(end_points['soft_prediction'] * labels, axis=1), name='soft_acc')
        reg_loss = tf.constant(0.0, dtype=tf.float32)
        if len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) > 0:
            reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='reg_Loss')
        total_loss = tf.add(pred_loss, reg_loss, name='total_loss')
    
    moving_stats = tf.train.ExponentialMovingAverage(decay=0.98)
    moving_stats_op = moving_stats.apply([pred_loss, reg_loss, total_loss, hard_accuracy, soft_accuracy])
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, moving_stats_op)

    tf.summary.scalar('pred_loss', moving_stats.average(pred_loss))
    tf.summary.scalar('reg_loss', moving_stats.average(reg_loss))
    tf.summary.scalar('total_loss', moving_stats.average(total_loss))
    tf.summary.scalar('hard_acc', moving_stats.average(hard_accuracy))
    tf.summary.scalar('soft_acc', moving_stats.average(soft_accuracy))
    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('slope_rate', slope_rate)
    if 'branch_result' in end_points:
        tf.summary.histogram('branch_result', end_points['branch_result'])
        tf.summary.histogram('branch_preact', end_points['branch_preact'])
        tf.summary.histogram('branch_soft', tf.nn.softmax(end_points['branch_preact'], dim=1))
    summary_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES), name='summary_op')
    
    update_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))

    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM_RATE, name='Momentum')
    gradients = optimizer.compute_gradients(total_loss)
    # with tf.variable_scope('gradient_clipping'):
    #     gradients = [(tf.clip_by_value(grad, -2.0, 2.0), var) for grad, var in gradients]
    grad_update_op = optimizer.apply_gradients(gradients, global_step=global_step)

    train_tensor = control_flow_ops.with_dependencies([update_op, grad_update_op], total_loss)
    
    return train_tensor, summary_op


def prepare_net_eval(images, labels):
    logits, end_points = resnet_v2_cifar10.resnet_v2_cifar(images, None, NET_SIZE_N, NUM_BRANCHES, is_training=False, reuse=False)
    pred = end_points['hard_prediction']
    labels = tf.squeeze(labels)

    # Define the metrics:
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'Accuracy': slim.metrics.streaming_accuracy(pred, labels),
        'Recall_5': slim.metrics.streaming_recall_at_k(
            logits, labels, 5),
        'Split': tf.metrics.mean_tensor(end_points['branch_result']),
    })
    for name, value in names_to_values.iteritems():
        summary_name = 'eval/%s' % name
        if value.shape.ndims == 0:
            op = tf.summary.scalar(summary_name, value, collections=[])
        else:
            op = tf.summary.histogram(summary_name, value, collections=[])
        op = tf.Print(op, [value], summary_name, summarize=1000)
        tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
    
    return names_to_updates

    
def main(_):
    if not FLAGS.dataset_dir or not FLAGS.model_log_dir:
        raise ValueError('Specify all flags')
    tf.logging.set_verbosity(tf.logging.INFO)

    dataset, batch_queue = prepare_dataset()
    train_t, summary_t = prepare_net(batch_queue, dataset.num_samples)

    slim.learning.train(
        train_t,
        number_of_steps=TOTAL_EPOCHS * (dataset.num_samples // BATCH_SIZE),
        logdir=FLAGS.model_log_dir,
        log_every_n_steps=100,
        # init_fn=load_pretrain_model(),
        summary_op=summary_t,
        save_summaries_secs=20,
        saver=tf.train.Saver(var_list=tf.model_variables() + [slim.get_or_create_global_step()], max_to_keep=20),
        save_interval_secs=600)
    
    tf.reset_default_graph()
    
    if tf.gfile.IsDirectory(FLAGS.model_log_dir):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.model_log_dir)
    else:
        checkpoint_path = FLAGS.model_log_dir
    
    dataset, images, labels = prepare_dataset_eval()
    names_to_updates = prepare_net_eval(images, labels)
    num_batches = math.ceil(dataset.num_samples / float(BATCH_SIZE_TEST))
    slim.evaluation.evaluate_once('',
        checkpoint_path=checkpoint_path,
        logdir=FLAGS.model_log_dir,
        num_evals=num_batches,
        eval_op=tf.group(*names_to_updates.values()),
        variables_to_restore=tf.model_variables() + [slim.get_or_create_global_step()])


if __name__ == '__main__':
  tf.app.run()
