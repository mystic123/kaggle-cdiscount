import io
import json
import time
from datetime import datetime
from multiprocessing import cpu_count

import bson
import numpy as np
import tensorflow as tf
from scipy import stats
from skimage.data import imread

from model import xception

THREADS = cpu_count()

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', './data', 'Directory with data files.')
tf.app.flags.DEFINE_string('model_dir', './logs/', 'Directory with saved checkpoint.')
tf.app.flags.DEFINE_string('out_file', './submission.csv', 'File with predictions.')
tf.app.flags.DEFINE_integer('batch_size', 1024, 'Batch size.')
tf.app.flags.DEFINE_integer('resize', 64, 'Batch size.')
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Whether to log device placement.')

IMG_WIDTH = 180
IMG_HEIGHT = 180
IMG_CHAN = 3

NUM_CLASSES = 5270

NUM_THREADS = 2
NUM_EXAMPLES = 1768182


def load_saved_model(sess, saver):
    saver.restore(sess, tf.train.latest_checkpoint(FLAGS.model_dir))


def main(argv=None):
    with tf.name_scope('input'):
        input = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, IMG_CHAN], name='img')
        if FLAGS.resize != IMG_WIDTH:
            input_resize = tf.image.resize_images(input, [FLAGS.resize, FLAGS.resize])
        else:
            input_resize = input
        input_resize = input_resize / 255.0
        input_resize = input_resize - 0.5
        input_resize = input_resize * 2.
        training_phase = tf.placeholder_with_default(False, [], name='training_phase')

        _, net = xception(input_resize, num_classes=5270, training_phase=training_phase)

    net_cls = tf.argmax(net, axis=1)

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    processed = set()
    # with open('./submission.csv_bak', 'r') as f:
    #     i = 0
    #     for line in f:
    #         if i < 1:
    #             i += 1
    #         else:
    #             prod_id = int(line.split(',')[0])
    #             processed.add(prod_id)

    with open(FLAGS.out_file, 'w') as out_file:
        out_file.write('_id,category_id\n')
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=FLAGS.log_device_placement)
        with tf.Session(config=config) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            load_saved_model(sess, saver)

            data = bson.decode_file_iter(open('PATH_TO_BSON_TEST_SET', 'rb'))

            with open('FILE_WITH_CATEGORIES_DICT', 'r') as f:
                label_to_cat = json.load(f)
            int_label_to_cat = {int(k): v for k, v in label_to_cat.items()}

            batch_imgs = []
            batch_prods = []
            step = 0
            for i, d in enumerate(data):
                prod_id = int(d['_id'])
                if prod_id in processed:
                    continue
                for e, pic in enumerate(d['imgs']):
                    pic = imread(io.BytesIO(pic['picture']))
                    batch_imgs.append(np.array(pic))
                    batch_prods.append(prod_id)

                if len(batch_imgs) >= FLAGS.batch_size:
                    feed_dict = {
                        input: batch_imgs,
                        training_phase: False
                    }

                    dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    t0 = time.time()
                    probs, predictions = sess.run([net, net_cls], feed_dict=feed_dict)
                    duration = time.time() - t0

                    labels = predictions
                    prod_to_labels = {}
                    for prod, label in zip(batch_prods, labels):
                        prod = int(prod)
                        label = int(label)
                        if prod not in prod_to_labels:
                            prod_to_labels[prod] = []
                        prod_to_labels[prod].append(label)

                    uniq_prods = set(prod_to_labels.keys())
                    sorted_prods = sorted(uniq_prods)

                    for p in sorted_prods:
                        label = stats.mode(prod_to_labels[p])[0][0]
                        cls = int_label_to_cat[label]
                        out_file.write('{},{}\n'.format(p, cls))
                        step += 1

                    format_str = '{}: prod {} / {} | dur: {:.2f}'
                    print(format_str.format(dt, step, NUM_EXAMPLES, duration))
                    batch_imgs = []
                    batch_prods = []

            feed_dict = {
                input: batch_imgs,
                training_phase: False
            }

            dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            t0 = time.time()
            predictions = sess.run(net_cls, feed_dict=feed_dict)
            step += 1
            duration = time.time() - t0

            labels = predictions
            prod_to_labels = {}
            for prod, label in zip(batch_prods, labels):
                prod = int(prod)
                label = int(label)
                if prod not in prod_to_labels:
                    prod_to_labels[prod] = []
                prod_to_labels[prod].append(label)

            uniq_prods = set([int(x) for x in batch_prods])
            uniq_prods = sorted(uniq_prods)

            for p in uniq_prods:
                label = stats.mode(prod_to_labels[p])[0][0]
                cls = int_label_to_cat[label]
                out_file.write('{},{}\n'.format(p, cls))
                step += 1

            format_str = '{}: prod {} / {} | dur: {:.2f}'
            print(format_str.format(dt, step, NUM_EXAMPLES, duration))


if __name__ == '__main__':
    tf.app.run()
