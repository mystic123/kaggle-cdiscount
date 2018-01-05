import numpy as np
import tensorflow as tf

_BATCH_NORM_DECAY = 0.9997
_BATCH_NORM_EPSILON = 0.001


def batch_norm_relu(inputs, training_phase, name):
    inputs = tf.layers.batch_normalization(inputs, axis=1, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
                                           training=training_phase, fused=True, name=name)
    inputs = tf.nn.relu(inputs)
    return inputs


def fixed_padding(inputs, kernel_size):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
    return padded_inputs


def conv_2d_pad(inputs, filters, kernel_size, strides, name):
    with tf.variable_scope(name):
        if strides > 1:
            inputs = fixed_padding(inputs, kernel_size)

        inputs = tf.layers.conv2d(inputs, filters, kernel_size, strides, padding=('SAME' if strides == 1 else 'VALID'),
                                  use_bias=False, kernel_initializer=tf.variance_scaling_initializer(),
                                  data_format='channels_first', name='conv')
        return inputs


def bottleneck_block(inputs, filters, strides, projection, training_phase, name):
    with tf.variable_scope(name):
        shortcut = inputs
        inputs = batch_norm_relu(inputs, training_phase, 'bn1')

        if projection:
            shortcut = conv_2d_pad(shortcut, 4 * filters, kernel_size=1, strides=strides, name='projection')

        inputs = conv_2d_pad(inputs, filters, kernel_size=1, strides=1, name='conv1')
        inputs = batch_norm_relu(inputs, training_phase, name='bn2')
        inputs = conv_2d_pad(inputs, filters, kernel_size=3, strides=strides, name='conv2')
        inputs = batch_norm_relu(inputs, training_phase, name='bn3')
        inputs = conv_2d_pad(inputs, filters=4 * filters, kernel_size=1, strides=1, name='conv3')

        return inputs + shortcut


def block_layer(inputs, filters, blocks, strides, training_phase, name):
    with tf.variable_scope(name):
        inputs = bottleneck_block(inputs, filters, strides=strides, projection=True,
                                  training_phase=training_phase, name='bottleneck_1')

        for i in range(1, blocks):
            inputs = bottleneck_block(inputs, filters, strides=1, projection=False,
                                      training_phase=training_phase, name='bottleneck_{}'.format(i + 1))

        return tf.identity(inputs, name)


def get_model_resnet(inputs, training_phase, num_classes, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()

    # Convert from channels_last (NHWC) to channels_first (NCHW). This
    # provides a large performance boost on GPU.
    inputs = tf.transpose(inputs, [0, 3, 1, 2])

    print(inputs.get_shape())
    inputs = conv_2d_pad(inputs, filters=64, kernel_size=7, strides=2, name='conv1')
    inputs = tf.identity(inputs, 'initial_conv')
    print(inputs.get_shape())

    inputs = tf.layers.max_pooling2d(inputs, pool_size=3, strides=2, padding='SAME', data_format='channels_first')
    inputs = tf.identity(inputs, 'initial_max_pool')
    print(inputs.get_shape())

    inputs = block_layer(inputs, filters=64, blocks=3, strides=1, training_phase=training_phase, name='block_1')
    print(inputs.get_shape())
    inputs = block_layer(inputs, filters=128, blocks=4, strides=2, training_phase=training_phase, name='block_2')
    print(inputs.get_shape())
    inputs = block_layer(inputs, filters=256, blocks=6, strides=2, training_phase=training_phase, name='block_3')
    print(inputs.get_shape())
    inputs = block_layer(inputs, filters=512, blocks=3, strides=2, training_phase=training_phase, name='block_4')
    print(inputs.get_shape())

    inputs = batch_norm_relu(inputs, training_phase, name='bn')
    inputs = tf.layers.average_pooling2d(inputs, pool_size=5, strides=1, padding='VALID', data_format='channels_first')
    inputs = tf.identity(inputs, 'final_avg_pool')
    print(inputs.get_shape())

    shape = inputs.get_shape().as_list()
    inputs = tf.reshape(inputs, [-1, np.prod(shape[1:])])
    print(inputs.get_shape())
    inputs = tf.layers.dense(inputs, num_classes)
    inputs = tf.identity(inputs, 'final_dense')
    print(inputs.get_shape())

    return inputs, tf.nn.softmax(inputs)


def vgg16_block(inputs, filters, name, training_phase=False):
    with tf.variable_scope(name):
        inputs = tf.layers.conv2d(inputs, filters, 3, 1, padding='SAME', use_bias=False,
                                  kernel_initializer=tf.variance_scaling_initializer(), data_format='channels_first',
                                  activation=tf.nn.relu, name='conv1')
        inputs = tf.layers.conv2d(inputs, filters, 3, 1, padding='SAME', use_bias=False,
                                  kernel_initializer=tf.variance_scaling_initializer(), data_format='channels_first',
                                  activation=tf.nn.relu, name='conv2')
        inputs = tf.layers.max_pooling2d(inputs, 2, 2, data_format='channels_first', name='max_pool')
        return inputs


def get_model_vgg16(inputs, training_phase, num_classes, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()

    # Convert from channels_last (NHWC) to channels_first (NCHW). This
    # provides a large performance boost on GPU.
    inputs = tf.transpose(inputs, [0, 3, 1, 2])
    print(inputs.get_shape())
    net = vgg16_block(inputs, 64, 'block1')
    print(net.get_shape())
    net = vgg16_block(net, 128, 'block2')
    print(net.get_shape())
    net = vgg16_block(net, 256, 'block3')
    print(net.get_shape())
    net = vgg16_block(net, 512, 'block4')
    print(net.get_shape())
    net = vgg16_block(net, 512, 'block5')

    shape = net.get_shape().as_list()
    net = tf.reshape(net, [-1, np.prod(shape[1:])])
    net = tf.layers.dropout(net, training=training_phase)
    net = tf.layers.dense(net, 4096, activation=tf.nn.relu, use_bias=False, name='dense1')
    net = tf.layers.dropout(net, training=training_phase)
    net = tf.layers.dense(net, 4096, activation=tf.nn.relu, use_bias=False, name='dense2')
    print(net.get_shape())
    net = tf.layers.dropout(net, training=training_phase)
    print(net.get_shape())
    net = tf.layers.dense(net, num_classes, use_bias=False, name='out')
    print(net.get_shape())

    return net, tf.nn.softmax(net)


# def conv_2d(inputs, filters, kernel_size, strides, name='conv'):
#     inputs = tf.layers.conv2d(inputs, filters, kernel_size, strides, padding='SAME',
#                               use_bias=False, kernel_initializer=tf.variance_scaling_initializer(1.43),
#                               data_format='channels_first', name=name)
#     return inputs
#
#
# def separable_conv_2d(inputs, filters, kernel_size, strides):
#     inputs = tf.layers.separable_conv2d(inputs, filters, kernel_size, strides, padding='SAME',
#                                         use_bias=False, depthwise_initializer=tf.variance_scaling_initializer(),
#                                         pointwise_initializer=tf.variance_scaling_initializer(),
#                                         data_format='channels_first', name='sep_conv')
#     return inputs
#
#
# def conv_bn(func, inputs, filters, kernel_size, strides, training_phase, name):
#     with tf.variable_scope(name):
#         inputs = func(inputs, filters, kernel_size, strides)
#         inputs = tf.layers.batch_normalization(inputs, axis=1, momentum=0.9997, training=training_phase, fused=True, name='bn')
#         return inputs


def entry_flow(inputs, training_phase, activation=tf.nn.relu):
    with tf.variable_scope('entry_flow'):
        inputs = conv_bn(conv_2d, inputs, 32, 3, 2, training_phase, name='conv_1')
        inputs = activation(inputs, name='act1')
        inputs = conv_bn(conv_2d, inputs, 64, 3, 1, training_phase, name='conv_2')
        inputs = activation(inputs, name='act')
        print(inputs.get_shape())

        residual = conv_bn(conv_2d, inputs, 128, 1, 2, training_phase, name='residual1')

        inputs = conv_bn(separable_conv_2d, inputs, 128, 3, 1, training_phase, name='sep_conv1')
        inputs = activation(inputs, name='act2')
        inputs = conv_bn(separable_conv_2d, inputs, 128, 3, 1, training_phase, name='sep_conv2')
        inputs = tf.layers.max_pooling2d(inputs, 3, 2, padding='SAME', data_format='channels_first')
        print(inputs.get_shape())

        inputs = inputs + residual

        residual = conv_bn(conv_2d, inputs, 256, 1, 2, training_phase, name='residual2')

        inputs = activation(inputs, name='act3')
        inputs = conv_bn(separable_conv_2d, inputs, 256, 3, 1, training_phase, name='sep_conv3')
        inputs = activation(inputs, name='act4')
        inputs = conv_bn(separable_conv_2d, inputs, 256, 3, 1, training_phase, name='sep_conv4')
        inputs = tf.layers.max_pooling2d(inputs, 3, 2, padding='SAME', data_format='channels_first')
        print(inputs.get_shape())

        inputs = inputs + residual

        residual = conv_bn(conv_2d, inputs, 728, 1, 2, training_phase, name='residual3')

        inputs = activation(inputs, name='act5')
        inputs = conv_bn(separable_conv_2d, inputs, 728, 3, 1, training_phase, name='sep_conv5')
        inputs = activation(inputs, name='act6')
        inputs = conv_bn(separable_conv_2d, inputs, 728, 3, 1, training_phase, name='sep_conv6')
        inputs = tf.layers.max_pooling2d(inputs, 3, 2, padding='SAME', data_format='channels_first')
        print(inputs.get_shape())

        inputs = inputs + residual

        return inputs


def middle_flow(inputs, training_phase, activation=tf.nn.relu, num_blocks=8):
    with tf.variable_scope('middle_flow'):
        for i in range(num_blocks):
            with tf.variable_scope('block_{}'.format(i)):
                residual = inputs
                inputs = activation(inputs, name='act1')
                inputs = conv_bn(separable_conv_2d, inputs, 728, 3, 1, training_phase, name='sep_conv1')
                inputs = activation(inputs, name='act2')
                inputs = conv_bn(separable_conv_2d, inputs, 728, 3, 1, training_phase, name='sep_conv2')
                inputs = activation(inputs, name='act3')
                inputs = conv_bn(separable_conv_2d, inputs, 728, 3, 1, training_phase, name='sep_conv3')
                inputs = inputs + residual
        return inputs


def exit_flow(inputs, training_phase, activation=tf.nn.relu):
    with tf.variable_scope('exit_flow'):
        residual = conv_bn(conv_2d, inputs, 1024, 1, 2, training_phase, name='residual')

        inputs = activation(inputs, name='act1')
        inputs = conv_bn(separable_conv_2d, inputs, 728, 3, 1, training_phase, name='sep_conv1')
        inputs = activation(inputs, name='act2')
        inputs = conv_bn(separable_conv_2d, inputs, 1024, 3, 1, training_phase, name='sep_conv2')
        inputs = tf.layers.max_pooling2d(inputs, 3, 2, padding='SAME', data_format='channels_first')

        inputs = inputs + residual

        inputs = conv_bn(separable_conv_2d, inputs, 1536, 3, 1, training_phase, name='sep_conv3')
        inputs = activation(inputs, name='act3')
        inputs = conv_bn(separable_conv_2d, inputs, 2048, 3, 1, training_phase, name='sep_conv4')
        inputs = activation(inputs, name='act4')
        print(inputs.get_shape())

        # Global Average Pooling
        inputs = tf.layers.average_pooling2d(inputs, 2, 2, padding='SAME', data_format='channels_first',
                                             name='avg_pool')
        return inputs


def get_model_xception(inputs, training_phase, num_classes, num_middle_blocks=8, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()

    # Convert from channels_last (NHWC) to channels_first (NCHW). This
    # provides a large performance boost on GPU.
    inputs = tf.transpose(inputs, [0, 3, 1, 2])
    print(inputs.get_shape())
    inputs = entry_flow(inputs, training_phase)
    print(inputs.get_shape())
    inputs = middle_flow(inputs, training_phase, num_blocks=num_middle_blocks)
    print(inputs.get_shape())
    inputs = exit_flow(inputs, training_phase)
    print(inputs.get_shape())
    inputs = conv_2d(inputs, 2048, 1, 1, name='conv_out1')
    print(inputs.get_shape())
    inputs = conv_2d(inputs, num_classes, 1, 1, name='conv_out2')
    print(inputs.get_shape())

    # inputs = tf.layers.dropout(inputs, training=training_phase)
    # inputs = tf.layers.dense(inputs, num_classes, use_bias=False,
    #                          kernel_initializer=tf.variance_scaling_initializer(), name='out')
    inputs = tf.reshape(inputs, [-1, np.prod(inputs.get_shape().as_list()[1:])])
    print(inputs.get_shape())

    return inputs, tf.nn.softmax(inputs)


def conv_2d(inputs, filters, kernel_size, strides, name='conv'):
    inputs = tf.layers.conv2d(inputs, filters, kernel_size, strides, padding='SAME',
                              use_bias=False, kernel_initializer=tf.variance_scaling_initializer(1.43),
                              data_format='channels_first', name=name)
    return inputs


def separable_conv_2d(inputs, filters, kernel_size, strides):
    inputs = tf.layers.separable_conv2d(inputs, filters, kernel_size, strides, padding='SAME',
                                        use_bias=False, depthwise_initializer=tf.variance_scaling_initializer(1.43),
                                        pointwise_initializer=tf.variance_scaling_initializer(),
                                        data_format='channels_first', name='sep_conv')
    return inputs


def conv_bn(func, inputs, filters, kernel_size, strides, training_phase, name):
    with tf.variable_scope(name):
        inputs = func(inputs, filters, kernel_size, strides)
        inputs = tf.layers.batch_normalization(inputs, axis=1, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
                                               training=training_phase, scale=False, fused=True, name='bn')
        return inputs


def xception(inputs, training_phase, num_classes, activation=tf.nn.elu, num_middle_blocks=8, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()

    inputs = tf.transpose(inputs, [0, 3, 1, 2], name='transpose_channels')
    with tf.variable_scope('Xception'):
        with tf.name_scope('entry'):
            inputs = conv_bn(conv_2d, inputs, 32, 3, 2, training_phase, name='conv_1')
            inputs = activation(inputs, name='act1')
            inputs = conv_bn(conv_2d, inputs, 64, 3, 1, training_phase, name='conv_2')
            inputs = activation(inputs, name='act2')
            print(inputs.get_shape())

            residual = conv_bn(conv_2d, inputs, 128, 1, 2, training_phase, name='residual1')

            inputs = conv_bn(separable_conv_2d, inputs, 128, 3, 1, training_phase, name='sep_conv1')
            inputs = activation(inputs, name='act3')
            inputs = conv_bn(separable_conv_2d, inputs, 128, 3, 1, training_phase, name='sep_conv2')
            inputs = tf.layers.max_pooling2d(inputs, 3, 2, padding='SAME', data_format='channels_first')
            print(inputs.get_shape())

            inputs = inputs + residual

            residual = conv_bn(conv_2d, inputs, 256, 1, 2, training_phase, name='residual2')

            inputs = activation(inputs, name='act4')
            inputs = conv_bn(separable_conv_2d, inputs, 256, 3, 1, training_phase, name='sep_conv3')
            inputs = activation(inputs, name='act5')
            inputs = conv_bn(separable_conv_2d, inputs, 256, 3, 1, training_phase, name='sep_conv4')
            inputs = tf.layers.max_pooling2d(inputs, 3, 2, padding='SAME', data_format='channels_first')
            print(inputs.get_shape())

            inputs = inputs + residual

            residual = conv_bn(conv_2d, inputs, 728, 1, 2, training_phase, name='residual3')

            inputs = activation(inputs, name='act6')
            inputs = conv_bn(separable_conv_2d, inputs, 728, 3, 1, training_phase, name='sep_conv5')
            inputs = activation(inputs, name='act7')
            inputs = conv_bn(separable_conv_2d, inputs, 728, 3, 1, training_phase, name='sep_conv6')
            inputs = tf.layers.max_pooling2d(inputs, 3, 2, padding='SAME', data_format='channels_first')
            print(inputs.get_shape())

            inputs = inputs + residual

        with tf.variable_scope('middle'):
            for i in range(num_middle_blocks):
                with tf.variable_scope('block_{}'.format(i)):
                    residual = inputs
                    inputs = activation(inputs, name='act1_{}'.format(i))
                    inputs = conv_bn(separable_conv_2d, inputs, 728, 3, 1, training_phase,
                                     name='sep_conv1_{}'.format(i))
                    inputs = activation(inputs, name='act2_{}'.format(i))
                    inputs = conv_bn(separable_conv_2d, inputs, 728, 3, 1, training_phase,
                                     name='sep_conv2_{}'.format(i))
                    inputs = activation(inputs, name='act3_{}'.format(i))
                    inputs = conv_bn(separable_conv_2d, inputs, 728, 3, 1, training_phase,
                                     name='sep_conv3_{}'.format(i))
                    inputs = inputs + residual

        with tf.variable_scope('exit'):
            residual = conv_bn(conv_2d, inputs, 1024, 1, 2, training_phase, name='residual')

            inputs = activation(inputs, name='act8')
            inputs = conv_bn(separable_conv_2d, inputs, 728, 3, 1, training_phase, name='sep_conv1')
            inputs = activation(inputs, name='act9')
            inputs = conv_bn(separable_conv_2d, inputs, 1024, 3, 1, training_phase, name='sep_conv2')
            inputs = tf.layers.max_pooling2d(inputs, 3, 2, padding='SAME', data_format='channels_first')

            inputs = inputs + residual

            inputs = conv_bn(separable_conv_2d, inputs, 1536, 3, 1, training_phase, name='sep_conv_3')
            inputs = activation(inputs, name='act10')
            inputs = conv_bn(separable_conv_2d, inputs, 2048, 3, 1, training_phase, name='sep_conv4')
            inputs = activation(inputs, name='act11')
            print(inputs.get_shape())

            inputs = tf.layers.average_pooling2d(inputs, 2, 2, padding='SAME', data_format='channels_first',
                                                 name='avg_pool')

            inputs = tf.reshape(inputs, [-1, np.prod(inputs.get_shape().as_list()[1:])])
            # inputs = tf.layers.conv2d(inputs, 2048, 1, name='conv_3')
            # inputs = tf.layers.conv2d(inputs, num_classes, 1, name='conv_4')
            # inputs = tf.reshape(inputs, [-1, np.prod(inputs.get_shape().as_list()[1:])])
            inputs = tf.layers.dropout(inputs, rate=0.2, training=training_phase)
            inputs = tf.layers.dense(inputs, num_classes, use_bias=False,
                                     kernel_initializer=tf.variance_scaling_initializer(), name='dense')

    return inputs, tf.nn.softmax(inputs)


def xception_bcnn(inputs, training_phase, num_classes, activation=tf.nn.elu, num_middle_blocks=8,
                  reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()

    inputs = tf.transpose(inputs, [0, 3, 1, 2], name='transpose_channels')
    with tf.variable_scope('Xception'):
        with tf.name_scope('entry'):
            inputs = conv_bn(conv_2d, inputs, 32, 3, 2, training_phase, name='conv_1')
            inputs = activation(inputs, name='act1')
            inputs = conv_bn(conv_2d, inputs, 64, 3, 1, training_phase, name='conv_2')
            inputs = activation(inputs, name='act2')
            print(inputs.get_shape())

            residual = conv_bn(conv_2d, inputs, 128, 1, 2, training_phase, name='residual1')

            inputs = conv_bn(separable_conv_2d, inputs, 128, 3, 1, training_phase, name='sep_conv1')
            inputs = activation(inputs, name='act3')
            inputs = conv_bn(separable_conv_2d, inputs, 128, 3, 1, training_phase, name='sep_conv2')
            inputs = tf.layers.max_pooling2d(inputs, 3, 2, padding='SAME', data_format='channels_first')
            print(inputs.get_shape())

            inputs = inputs + residual

            residual = conv_bn(conv_2d, inputs, 256, 1, 2, training_phase, name='residual2')

            inputs = activation(inputs, name='act4')
            inputs = conv_bn(separable_conv_2d, inputs, 256, 3, 1, training_phase, name='sep_conv3')
            inputs = activation(inputs, name='act5')
            inputs = conv_bn(separable_conv_2d, inputs, 256, 3, 1, training_phase, name='sep_conv4')
            inputs = tf.layers.max_pooling2d(inputs, 3, 2, padding='SAME', data_format='channels_first')
            print(inputs.get_shape())

            inputs = inputs + residual

            residual = conv_bn(conv_2d, inputs, 728, 1, 2, training_phase, name='residual3')

            inputs = activation(inputs, name='act6')
            inputs = conv_bn(separable_conv_2d, inputs, 728, 3, 1, training_phase, name='sep_conv5')
            inputs = activation(inputs, name='act7')
            inputs = conv_bn(separable_conv_2d, inputs, 728, 3, 1, training_phase, name='sep_conv6')
            inputs = tf.layers.max_pooling2d(inputs, 3, 2, padding='SAME', data_format='channels_first')
            print(inputs.get_shape())

            inputs = inputs + residual

            level1_logits = inputs
            level1_logits = tf.reshape(level1_logits, [-1, np.prod(level1_logits.get_shape().as_list()[1:])])
            level1_logits = tf.layers.dropout(level1_logits, training=training_phase)
            level1_logits = tf.layers.dense(level1_logits, num_classes[0], use_bias=False,
                                            kernel_initializer=tf.variance_scaling_initializer(), name='lvl1_dense')

        with tf.variable_scope('middle'):
            for i in range(num_middle_blocks):
                with tf.variable_scope('block_{}'.format(i)):
                    residual = inputs
                    inputs = activation(inputs, name='act1_{}'.format(i))
                    inputs = conv_bn(separable_conv_2d, inputs, 728, 3, 1, training_phase,
                                     name='sep_conv1_{}'.format(i))
                    inputs = activation(inputs, name='act2_{}'.format(i))
                    inputs = conv_bn(separable_conv_2d, inputs, 728, 3, 1, training_phase,
                                     name='sep_conv2_{}'.format(i))
                    inputs = activation(inputs, name='act3_{}'.format(i))
                    inputs = conv_bn(separable_conv_2d, inputs, 728, 3, 1, training_phase,
                                     name='sep_conv3_{}'.format(i))
                    inputs = inputs + residual

            level2_logits = inputs
            level2_logits = tf.reshape(level2_logits, [-1, np.prod(level2_logits.get_shape().as_list()[1:])])
            level2_logits = tf.layers.dropout(level2_logits, training=training_phase)
            lvl1_lvl2_concat = tf.concat([level1_logits, level2_logits], axis=1, name='lvl1_lvl2_concat')
            level2_logits = tf.layers.dense(lvl1_lvl2_concat, num_classes[1], use_bias=False,
                                            kernel_initializer=tf.variance_scaling_initializer(), name='lvl2_dense')

        with tf.variable_scope('exit'):
            residual = conv_bn(conv_2d, inputs, 1024, 1, 2, training_phase, name='residual')

            inputs = activation(inputs, name='act8')
            inputs = conv_bn(separable_conv_2d, inputs, 728, 3, 1, training_phase, name='sep_conv1')
            inputs = activation(inputs, name='act9')
            inputs = conv_bn(separable_conv_2d, inputs, 1024, 3, 1, training_phase, name='sep_conv2')
            inputs = tf.layers.max_pooling2d(inputs, 3, 2, padding='SAME', data_format='channels_first')

            inputs = inputs + residual

            inputs = conv_bn(separable_conv_2d, inputs, 1536, 3, 1, training_phase, name='sep_conv3')
            inputs = activation(inputs, name='act10')
            inputs = conv_bn(separable_conv_2d, inputs, 2048, 3, 1, training_phase, name='sep_conv4')
            inputs = activation(inputs, name='act11')
            print(inputs.get_shape())

            inputs = tf.layers.average_pooling2d(inputs, 2, 2, padding='SAME', data_format='channels_first',
                                                 name='avg_pool')

            inputs = tf.reshape(inputs, [-1, np.prod(inputs.get_shape().as_list()[1:])])
            # inputs = tf.layers.conv2d(inputs, 2048, 1, name='conv_3')
            # inputs = tf.layers.conv2d(inputs, num_classes, 1, name='conv_4')
            # inputs = tf.reshape(inputs, [-1, np.prod(inputs.get_shape().as_list()[1:])])
            inputs = tf.layers.dropout(inputs, training=training_phase)
            lvl2_lvl3_concat = tf.concat([level2_logits, inputs], axis=1, name='lvl2_lvl3_concat')
            inputs = tf.layers.dense(lvl2_lvl3_concat, num_classes[2], use_bias=False,
                                     kernel_initializer=tf.variance_scaling_initializer(), name='lvl3_dense')

    return (level1_logits, level2_logits, inputs), tf.nn.softmax(inputs)
