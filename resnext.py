import tensorflow as tf

_BATCH_NORM_DECAY = 0.9997
_BATCH_NORM_EPSILON = 0.001


def batch_norm(inputs, training_phase, name):
    inputs = tf.layers.batch_normalization(inputs, axis=1, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
                                           training=training_phase, fused=True, name=name)
    return inputs


def conv_2d(inputs, filters, kernel_size, strides=1, padding='SAME', name='conv'):
    inputs = tf.layers.conv2d(inputs, filters, kernel_size, strides, padding=padding, use_bias=False,
                              kernel_initializer=tf.variance_scaling_initializer(1.43), data_format='channels_first',
                              name=name)
    return inputs


def resnext_block(inputs, out_filters, filters, training_phase, downsample=False, cardinality=32, name='block'):
    with tf.variable_scope(name):
        residual = inputs
        residual = conv_2d(residual, out_filters, 1, strides=2 if downsample else 1, name='projection')

        groups = []

        for i in range(cardinality):
            with tf.variable_scope('branch_{}'.format(i)):
                conv1 = conv_2d(inputs, filters, 1, name='conv1')
                conv1 = batch_norm(conv1, training_phase, name='bn1')
                conv1 = tf.nn.elu(conv1)

                conv2 = conv_2d(conv1, filters, 3, strides=2 if downsample else 1, name='conv2')
                conv2 = batch_norm(conv2, training_phase, name='bn2')
                conv2 = tf.nn.elu(conv2)
                groups.append(conv2)

        concat = tf.concat(groups, axis=1)
        conv = conv_2d(concat, out_filters, 1, name='conv')
        conv = batch_norm(conv, training_phase, name='bn')

        inputs = residual + conv
        return tf.nn.elu(inputs)


def resnext(inputs, training_phase, num_classes, num_blocks=(3, 4, 6, 3), reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()

    # Convert from channels_last (NHWC) to channels_first (NCHW). This
    # provides a large performance boost on GPU.
    inputs = tf.transpose(inputs, [0, 3, 1, 2])
    print(inputs.get_shape())

    inputs = conv_2d(inputs, 64, kernel_size=7, strides=2, name='conv1')
    inputs = batch_norm(inputs, training_phase, name='bn1')
    inputs = tf.nn.elu(inputs)
    print(inputs.get_shape())
    inputs = tf.layers.max_pooling2d(inputs, 3, 2, padding='SAME', data_format='channels_first', name='max_pool1')
    print(inputs.get_shape())
    for i in range(num_blocks[0]):
        inputs = resnext_block(inputs, 256, 4, training_phase, downsample=(i == 0), name='block_0_{}'.format(i))
    print(inputs.get_shape())
    for i in range(num_blocks[1]):
        inputs = resnext_block(inputs, 512, 8, training_phase, downsample=(i == 0), name='block_1_{}'.format(i))
    print(inputs.get_shape())
    for i in range(num_blocks[2]):
        inputs = resnext_block(inputs, 1024, 16, training_phase, downsample=(i == 0), name='block_2_{}'.format(i))
    print(inputs.get_shape())
    for i in range(num_blocks[3]):
        inputs = resnext_block(inputs, 2048, 32, training_phase, downsample=(i == 0), name='block_4_{}'.format(i))
    print(inputs.get_shape())
    inputs = tf.reduce_mean(inputs, axis=[2, 3])
    print(inputs.get_shape())

    inputs = tf.layers.dense(inputs, num_classes, use_bias=False, name='fc')
    print(inputs.get_shape())
    return inputs, tf.nn.softmax(inputs)


if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [None, 180, 180, 3], 'inputs')
    net = resnext(inputs=x, training_phase=False, num_classes=5270)
