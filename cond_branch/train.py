from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.ops import control_flow_ops
import imagenet
import inception_v2
import preprocess_inception

slim = tf.contrib.slim

BATCH_SIZE=128
NUM_BRANCHES=2
INITIAL_LEARNING_RATE=0.0 # 1e-4
DECAY_RATE=0.8
INFLAT_RATE=1.1
MOMENTUM_RATE=0.9

DISK_READER=4
PREPROCESSOR=4
TRAIN_IMAGE_SIZE=224

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_string(
    'model_log_dir', None,
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_string(
    'pretrain_checkpoint', None,
    'The path to a checkpoint from which to fine-tune.')

FLAGS = tf.app.flags.FLAGS


def prepare_dataset():
    with tf.device('/cpu:0'):
        with tf.variable_scope('training_data_provider'):
            dataset = imagenet.get_split('train', FLAGS.dataset_dir)

            provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset,
                num_readers=DISK_READER,
                common_queue_capacity=20 * BATCH_SIZE,
                common_queue_min=10 * BATCH_SIZE)
            [image, label] = provider.get(['image', 'label'])

            image = preprocess_inception.preprocess_image(image, TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE, is_training=True)

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

def prepare_net(batch_queue, num_samples):
    global_step = slim.create_global_step()

    batch_num = num_samples // BATCH_SIZE
    learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, 2 * batch_num, DECAY_RATE, staircase=True, name='learning_rate')
    slope_rate = tf.train.exponential_decay(1.0, global_step, batch_num, INFLAT_RATE, staircase=True, name='slope_rate')

    images, labels = batch_queue.dequeue()
    logits, end_points = inception_v2.inception_v2(images, slope_rate, NUM_BRANCHES, is_training=False, reuse=False)

    with tf.variable_scope('stats'):
        pred_loss = tf.losses.softmax_cross_entropy(labels, logits, scope='cross_entropy_loss')
        hard_accuracy = tf.reduce_mean(tf.to_float(tf.equal(end_points['hard_prediction'], tf.argmax(labels, axis=1))), name='hard_acc')
        soft_accuracy = tf.reduce_mean(tf.reduce_sum(end_points['soft_prediction'] * labels, axis=1), name='soft_acc')
        reg_loss = 0.0
        if len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) > 0:
            reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='reg_Loss')
        total_loss = tf.add(pred_loss, reg_loss, name='total_loss')
    tf.summary.scalar('pred_loss', pred_loss)
    tf.summary.scalar('reg_loss', reg_loss)
    tf.summary.scalar('total_loss', total_loss)
    tf.summary.scalar('hard_acc', hard_accuracy)
    tf.summary.scalar('soft_acc', soft_accuracy)
    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('slope_rate', slope_rate)
    tf.summary.histogram('branch_preact', end_points['branch_preact'])
    summary_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES), name='summary_op')
    
    update_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))

    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM_RATE, name='Momentum')
    gradients = optimizer.compute_gradients(total_loss)
    # with tf.variable_scope('gradient_clipping'):
    #     gradients = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gradients]
    grad_update_op = optimizer.apply_gradients(gradients, global_step=global_step)

    train_tensor = control_flow_ops.with_dependencies([update_op, grad_update_op], total_loss)
    
    return train_tensor, summary_op

    
def load_pretrain_model():
    variable_restoring = []
    for var in slim.get_model_variables():
        var_name = var.op.name
        var_name_comp = var_name.split('/')
        if 'CondBranchFn' in var_name_comp:
            continue
        var_name_comp = filter(lambda s: not s.startswith('CondBranch_'), var_name_comp)
        var_name_new = '/'.join(var_name_comp)
        added_var = False
        for m in variable_restoring:
            if var_name_new not in m:
                m[var_name_new] = var
                added_var = True
                break
        if not added_var:
            variable_restoring.append({var_name_new : var})
    tf.logging.info('Fine-tuning from %s' % FLAGS.pretrain_checkpoint)

    init_fn = []
    with tf.variable_scope('init_fn'):
        for m in variable_restoring:
            init_fn.append(slim.assign_from_checkpoint_fn(FLAGS.pretrain_checkpoint, m))
    def _init_fn(sess):
        for f in init_fn:
            f(sess)
    return _init_fn
        


def main(_):
    if not FLAGS.dataset_dir or not FLAGS.model_log_dir or not FLAGS.pretrain_checkpoint:
        raise ValueError('Specify all flags')
    tf.logging.set_verbosity(tf.logging.INFO)

    dataset, batch_queue = prepare_dataset()
    train_t, summary_t = prepare_net(batch_queue, dataset.num_samples)

    slim.learning.train(
        train_t,
        logdir=FLAGS.model_log_dir,
        log_every_n_steps=100,
        init_fn=load_pretrain_model(),
        summary_op=summary_t,
        save_summaries_secs=60,
        saver=tf.train.Saver(var_list=tf.model_variables() + [slim.get_or_create_global_step()], max_to_keep=20),
        save_interval_secs=1800)
    

if __name__ == '__main__':
  tf.app.run()
