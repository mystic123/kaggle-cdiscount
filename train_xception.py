import os
import time
from datetime import datetime
from multiprocessing import cpu_count

import tensorflow as tf
from queue import Queue

from model import xception
from utils import PrefetchingThread

THREADS = cpu_count()

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', './data', 'Directory with data files.')
tf.app.flags.DEFINE_string('train_dir', './logs/', 'Directory where to write event logs and checkpoint.')
tf.app.flags.DEFINE_string('model_dir', './logs/', 'Directory with saved checkpoint.')
tf.app.flags.DEFINE_integer('max_steps', 10000000, 'Number of batches to run.')
tf.app.flags.DEFINE_integer('batch_size', 512, 'Batch size.')
tf.app.flags.DEFINE_integer('num_gpus', 2, 'Num of GPU.')
tf.app.flags.DEFINE_integer('decay', 15, 'Number of epochs for decay step')
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Whether to log device placement.')
tf.app.flags.DEFINE_boolean('cont', False, 'Whether to continue training from checkpoint.')

INITIAL_LEARNING_RATE = 0.001
LEARNING_RATE_DECAY_FACTOR = 0.1

IMG_WIDTH = 180
IMG_HEIGHT = 180
IMG_CHAN = 3

NUM_CLASSES = 5270

MOMENTUM = 0.9
WEIGHT_DECAY = 1e-5

NUM_THREADS = 2
NUM_EXAMPLES = 12371293

PS_OPS = [
    'Variable', 'VariableV2', 'AutoReloadVariable', 'MutableHashTable',
    'MutableHashTableOfTensors', 'MutableDenseHashTable'
]


def assign_to_device(device, ps_device=None):
    """Returns a function to place variables on the ps_device.

    Args:
        device: Device for everything but variables
        ps_device: Device to put the variables on.  Example values are GPU:0 and
        CPU:0.

    If ps_device is not set then the variables will be placed on the device.
    The best device for shared varibles depends on the platform as well as the
    model.  Start with CPU:0 and then test GPU:0 to see if there is an
    improvement.

    """

    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return "/" + ps_device
        else:
            return device

    return _assign


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def load_saved_model(sess, saver):
    saver.restore(sess, tf.train.latest_checkpoint(FLAGS.model_dir))


def loss(logits, labels):
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    loss = cross_entropy + WEIGHT_DECAY * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    return loss


def accuracy(net, labels):
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(net, 1), labels)
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        acc = tf.reduce_mean(correct_prediction)
    return acc


def main(argv=None):
    prefetch_queue = Queue(maxsize=FLAGS.batch_size * 100)
    validation_queue = Queue(maxsize=FLAGS.batch_size * 10)

    for i in range(NUM_THREADS):
        prefetching_thread = PrefetchingThread(prefetch_queue, 'PATH_TO_TRAIN_DIR')
        prefetching_thread.start()

    validation_prefetch_thread = PrefetchingThread(validation_queue, 'PATH_TO_VALIDATION_DIR')
    validation_prefetch_thread.start()

    training_phase = tf.placeholder_with_default(False, [], name='training_phase')
    global_step = tf.Variable(0, name='global_step', trainable=False)

    num_steps_per_epoch = NUM_EXAMPLES // (FLAGS.batch_size * FLAGS.num_gpus)
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, FLAGS.decay * num_steps_per_epoch,
                                    LEARNING_RATE_DECAY_FACTOR, staircase=True)
    tf.summary.scalar('learning_rate', lr, collections=['train'])

    opt = tf.train.AdamOptimizer(lr)
    # grads = optim.compute_gradients(loss, tf.trainable_variables())
    # apply_gradient_op = optim.apply_gradients(grads, global_step=global_step)

    tower_grads = []
    inputs = []
    y_s = []
    accuracys = []
    cross_entropies = []
    losses = []
    reuse_variables = None
    for i in range(FLAGS.num_gpus):
        with tf.device(assign_to_device('/gpu:{}'.format(i), ps_device='gpu:0')):
            with tf.name_scope('{}_{}'.format('TOWER', i)) as n_scope:
                with tf.name_scope('input'):
                    input = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, IMG_CHAN], name='img')
                    inputs.append(input)
                if FLAGS.resize != IMG_WIDTH:
                    input_resize = tf.image.resize_images(input, [FLAGS.resize, FLAGS.resize])
                else:
                    input_resize = input
                input_resize = input_resize / 255.0
                input_resize = input_resize - 0.5
                input_resize = input_resize * 2.
                with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
                    y_ = tf.placeholder(tf.int64, [None])
                    y_s.append(y_)
                    logits, net = xception(input_resize, training_phase, NUM_CLASSES, num_middle_blocks=8,
                                           reuse=reuse_variables)
                    tf.summary.histogram('{}/logits'.format(n_scope), logits, collections=['train'])
                    tower_loss = loss(logits, y_)
                    acc = accuracy(net, y_)
                    accuracys.append(acc)
                    losses.append(tower_loss)
                    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=logits)
                    cross_entropies.append(cross_entropy)
                    grads = opt.compute_gradients(tower_loss)
                    tower_grads.append(grads)
                reuse_variables = True

    acc = tf.reduce_mean(accuracys)
    loss_ = tf.reduce_mean(losses)
    cross_entropy_ = tf.reduce_mean(cross_entropies)
    tf.summary.scalar('loss/train', loss_, collections=['train'])
    tf.summary.scalar('loss/valid', cross_entropy_, collections=['validation'])
    tf.summary.scalar('accuracy/train', acc, collections=['train'])
    tf.summary.scalar('accuracy/valid', acc, collections=['validation'])

    # grads = average_gradients(tower_grads)
    # for grad, var in grads:
    #     tf.summary.histogram(var.name + '/grads', grad, collections=['train'])
    #     tf.summary.histogram(var.op.name + '/weights', var, collections=['train'])
    #
    # variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    #     train_op = tf.no_op(name='train')

    # Batch norm requires update_ops to be added as a train_op dependency.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = opt.apply_gradients(grads, global_step=global_step)
    # train_op = apply_gradient_op

    train_summary_op = tf.summary.merge_all(key='train')
    valid_summary_op = tf.summary.merge_all(key='validation')

    if not tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.MakeDirs(FLAGS.train_dir)

    run_name = '{}'.format(datetime.now().strftime('%Y%m%d_%H%M%S'))
    run_dir = os.path.join(FLAGS.train_dir, run_name)

    if FLAGS.cont:
        run_dir = FLAGS.model_dir

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=FLAGS.log_device_placement)
    # config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        if FLAGS.cont:
            load_saved_model(sess, saver)

        global_step_val = sess.run(global_step)

        if FLAGS.cont:
            print('Loaded model, step:', global_step_val)

        summary_writer = tf.summary.FileWriter(run_dir, sess.graph)
        for step in range(global_step_val, FLAGS.max_steps):
            dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            feed_dict = {}
            t0 = time.time()
            for i in range(FLAGS.num_gpus):
                batch_imgs = []
                batch_labels = []
                for _ in range(FLAGS.batch_size):
                    (_, _, cls), img = prefetch_queue.get()
                    batch_imgs.append(img)
                    batch_labels.append(cls)
                feed_dict[inputs[i]] = batch_imgs
                feed_dict[y_s[i]] = batch_labels
                feed_dict[training_phase] = True
            fetch_time = time.time() - t0
            t1 = time.time()
            _, loss_val, acc_val = sess.run([train_op, loss_, acc], feed_dict=feed_dict)
            duration = time.time() - t1

            if step % 10 == 0:
                summary_str = sess.run(train_summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, global_step=step)
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = (num_examples_per_step / duration) * FLAGS.num_gpus
                sec_per_batch = duration

                format_str = '{}: step {} loss = {:.8f} acc = {:.4f} ({:.2f} ex/s; {:.2f} s/batch; fetch: {:.2f})'
                print(format_str.format(dt, step, loss_val, acc_val, examples_per_sec, sec_per_batch, fetch_time))

            if step % 100 == 0:
                feed_dict = {}
                t0 = time.time()
                for i in range(FLAGS.num_gpus):
                    batch_imgs = []
                    batch_labels = []
                    for _ in range(FLAGS.batch_size):
                        (_, _, cls), img = validation_queue.get()
                        batch_imgs.append(img)
                        batch_labels.append(cls)
                    feed_dict[inputs[i]] = batch_imgs
                    feed_dict[y_s[i]] = batch_labels
                    feed_dict[training_phase] = True
                fetch_time = time.time() - t0
                t1 = time.time()
                summary_str, val_loss, val_acc = sess.run([valid_summary_op, cross_entropy_, acc],
                                                          feed_dict=feed_dict)
                duration = time.time() - t1
                summary_writer.add_summary(summary_str, global_step=step)
                format_str = '{}: (VALID) step {} loss = {:.8f} acc = {:.4f} time: {:.2f} fetch: {:.2f}'
                print(format_str.format(dt, step, val_loss, val_acc, duration, fetch_time))

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(run_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=global_step)


if __name__ == '__main__':
    tf.app.run()
