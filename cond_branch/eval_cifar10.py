from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from tensorflow.python.ops import control_flow_ops
import cifar10
import preprocessing_cifar10
import resnet_v2_cifar10

slim = tf.contrib.slim

BATCH_SIZE=500
TRAIN_IMAGE_SIZE=32
DISK_READER=2
PREPROCESSOR=2
NUM_BRANCHES=1

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', None, 'Directory where the results are saved to.')

FLAGS = tf.app.flags.FLAGS

def prepare_dataset():
    with tf.device('/cpu:0'):
        with tf.variable_scope('evaluation_data_provider'):

            dataset = cifar10.get_split('test', FLAGS.dataset_dir)

            provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset, shuffle=False,
                num_readers=DISK_READER,
                common_queue_capacity=20 * BATCH_SIZE,
                common_queue_min=10 * BATCH_SIZE)
            [image, label] = provider.get(['image', 'label'])

            image = preprocessing_cifar10.preprocess_image(image, TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE, is_training=False)

            images, labels = tf.train.batch(
                [image, label],
                batch_size=BATCH_SIZE,
                num_threads=PREPROCESSOR,
                capacity=10 * BATCH_SIZE)

    return dataset, images, labels


def prepare_net(images, labels):
    logits, end_points = resnet_v2_cifar10.resnet_v2_cifar_no_branch(images, None, NUM_BRANCHES, is_training=False, reuse=False)
    pred = end_points['hard_prediction']
    labels = tf.squeeze(labels)

    # Define the metrics:
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'Accuracy': slim.metrics.streaming_accuracy(pred, labels),
        'Recall_5': slim.metrics.streaming_recall_at_k(
            logits, labels, 5),
    })
    for name, value in names_to_values.iteritems():
        summary_name = 'eval/%s' % name
        op = tf.summary.scalar(summary_name, value, collections=[])
        op = tf.Print(op, [value], summary_name)
        tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
    
    return names_to_updates


def main(_):
    if not FLAGS.dataset_dir or not FLAGS.checkpoint_path or not FLAGS.eval_dir:
        raise ValueError('Specify all flags')
    tf.logging.set_verbosity(tf.logging.INFO)

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path
    
    dataset, images, labels = prepare_dataset()
    names_to_updates = prepare_net(images, labels)

    num_batches = math.ceil(dataset.num_samples / float(BATCH_SIZE))

    slim.evaluation.evaluate_once('',
        checkpoint_path=checkpoint_path,
        logdir=FLAGS.eval_dir,
        num_evals=num_batches,
        eval_op=tf.group(*names_to_updates.values()),
        variables_to_restore=tf.model_variables() + [slim.get_or_create_global_step()])
    

if __name__ == '__main__':
  tf.app.run()
