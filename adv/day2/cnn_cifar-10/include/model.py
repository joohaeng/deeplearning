import tensorflow as tf
from math import sqrt


def model():
    _IMAGE_SIZE = 32
    _IMAGE_CHANNELS = 3
    _NUM_CLASSES = 10
    #_RESHAPE_SIZE = ?

    with tf.name_scope('data'):
        x = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE * _IMAGE_SIZE * _IMAGE_CHANNELS], name='Input')
        y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')
        x_image = tf.reshape(x, [-1, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS], name='images')

    def put_kernels_on_grid(kernel, pad=1):

        '''Visualize conv. features as an image (mostly for the 1st layer).
        Place kernel into a grid, with some paddings between adjacent filters.
        Args:
          kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
          (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                               User is responsible of how to break into two multiples.
          pad:               number of black pixels around each filter (between them)
        Return:
          Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
        '''

        # get shape of the grid. NumKernels == grid_Y * grid_X
        def factorization(n):
            for i in range(int(sqrt(float(n))), 0, -1):
                if n % i == 0:
                    if i == 1: print('Who would enter a prime number of filters')
                    return (i, int(n / i))

        (grid_Y, grid_X) = factorization(kernel.get_shape()[3].value)

        x_min = tf.reduce_min(kernel)
        x_max = tf.reduce_max(kernel)

        kernel1 = (kernel - x_min) / (x_max - x_min)

        # pad X and Y
        x1 = tf.pad(kernel1, tf.constant([[pad, pad], [pad, pad], [0, 0], [0, 0]]), mode='CONSTANT')

        # X and Y dimensions, w.r.t. padding
        Y = kernel1.get_shape()[0] + 2 * pad
        X = kernel1.get_shape()[1] + 2 * pad

        channels = kernel1.get_shape()[2]

        # put NumKernels to the 1st dimension
        x2 = tf.transpose(x1, (3, 0, 1, 2))
        # organize grid on Y axis
        x3 = tf.reshape(x2, tf.stack([grid_X, Y * grid_Y, X, channels]))

        # switch X and Y axes
        x4 = tf.transpose(x3, (0, 2, 1, 3))
        # organize grid on X axis
        x5 = tf.reshape(x4, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

        # back to normal order (not combining with the next step for clarity)
        x6 = tf.transpose(x5, (2, 1, 3, 0))

        # to tf.image_summary order [batch_size, height, width, channels],
        #   where in this case batch_size == 1
        x7 = tf.transpose(x6, (3, 0, 1, 2))

        # scaling to [0, 255] is not necessary for tensorboard
        return x7

    # 64 5x5 filters (kernels)
    with tf.variable_scope('conv1') as scope:
        kernel = tf.get_variable('weights', shape=[5, 5, 3, 64], initializer=tf.truncated_normal_initializer(stddev=5e-2))
        conv = tf.nn.conv2d(x_image, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    with tf.variable_scope('Visualization'):
        grid = put_kernels_on_grid(kernel)
        tf.summary.image('conv1/filters', grid, max_outputs=1)
    tf.summary.histogram('Convolution_layers/conv1', conv1)
    tf.summary.scalar('Convolution_layers/conv1', tf.nn.zero_fraction(conv1))

    norm1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    # 32x32x64
    pool1 = tf.nn.max_pool(norm1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1') 

    # 64 5x5 filters, output : conv2
    with tf.variable_scope('conv2') as scope:
        kernel = tf.get_variable('weights', shape=[5, 5, 64, 64], initializer=tf.truncated_normal_initializer(stddev=5e-2))
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)

    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # 128 3x3 filters, output : conv3
    with tf.variable_scope('conv3') as scope:
        kernel = tf.get_variable('weights', shape=[3, 3, 64, 128], initializer=tf.truncated_normal_initializer(stddev=5e-2))
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [128], initializer=tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name=scope.name)

    # 128 3x3 filters, output : conv4
    with tf.variable_scope('conv4') as scope:
        kernel = tf.get_variable('weights', shape=[3, 3, 128, 128], initializer=tf.truncated_normal_initializer(stddev=5e-2))
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [128], initializer=tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(pre_activation, name=scope.name)

    # 128 3x3 filters, output : conv5
    with tf.variable_scope('conv5') as scope:
        kernel = tf.get_variable('weights', shape=[3, 3, 128, 128], initializer=tf.truncated_normal_initializer(stddev=5e-2))
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [128], initializer=tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(pre_activation, name=scope.name)

    norm3 = tf.nn.lrn(conv5, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
    pool3 = tf.nn.max_pool(norm3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
    # [batch, ?, ?, 128]

    # FC: _RESHAPE_SIZE --> 384, output = local3
    with tf.variable_scope('fully_connected1') as scope:
        reshape = tf.reshape(pool3, [-1, pool3.get_shape()[1].value*pool3.get_shape()[2].value*128 ])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights', shape=[dim, 384], initializer=tf.truncated_normal_initializer(stddev=5e-2))
        biases = tf.get_variable('biases', [384], initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    tf.summary.histogram('Fully connected layers/fc1', local3)
    tf.summary.scalar('Fully connected layers/fc1', tf.nn.zero_fraction(local3))

    # FC: 384 --> 192, output = local4
    with tf.variable_scope('fully_connected2') as scope:
        weights = tf.get_variable('weights', shape=[384, 192], initializer=tf.truncated_normal_initializer(stddev=5e-2))
        biases = tf.get_variable('biases', [192], initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)

    # FC: 192 --> _NUM_CLASSES
    with tf.variable_scope('output') as scope:
        weights = tf.get_variable('weights', shape=[192, _NUM_CLASSES], initializer=tf.truncated_normal_initializer(stddev=1/192.0))
        biases = tf.get_variable('biases', [_NUM_CLASSES], initializer=tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    tf.summary.histogram('Fully connected layers/output', softmax_linear)

    global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
    y_pred_cls = tf.argmax(softmax_linear, dimension=1)

    return x, y, softmax_linear, global_step, y_pred_cls
