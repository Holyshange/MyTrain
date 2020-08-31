import tensorflow as tf

def create_variable(name, shape, initializer, dtype=tf.float32, trainable=True):
    return tf.get_variable(name, shape=shape, dtype=dtype, 
                           initializer=initializer, trainable=trainable)
    
def conv2d(inputs, num_filters, filter_size=1, strides=1, scope='conv'):
    inputs_shape = inputs.get_shape().as_list()
    in_channels = inputs_shape[-1]
    with tf.variable_scope(scope):
        weights = create_variable("Filter", 
                                  shape=[filter_size, filter_size, in_channels, num_filters], 
                                  initializer=tf.truncated_normal_initializer(stddev=0.01))
        return tf.nn.conv2d(inputs, weights, strides=[1, strides, strides, 1], padding="SAME")

def depthwise_conv2d(inputs, filter_size=3, channel_multiplier=1, strides=1, scope='dw_conv'):
    inputs_shape = inputs.get_shape().as_list()
    in_channels = inputs_shape[-1]
    with tf.variable_scope(scope):
        weights = create_variable("Filter", 
                                  shape=[filter_size, filter_size, in_channels, channel_multiplier], 
                                  initializer=tf.truncated_normal_initializer(stddev=0.01))
        return tf.nn.depthwise_conv2d(inputs, weights, strides=[1, strides, strides, 1], 
                                      padding="SAME", rate=[1, 1])

def bacth_norm(inputs, epsilon=1e-05, momentum=0.99, is_training=True, scope='bn'):
    with tf.variable_scope(scope):
        return tf.layers.batch_normalization(inputs, momentum=momentum, 
                                             epsilon=epsilon, training=is_training)

def relu(inputs, scope='relu'):
    with tf.variable_scope(scope):
        return tf.nn.relu(inputs)

def avg_pool(inputs, pool_size, scope='avg_pool'):
    with tf.variable_scope(scope):
        return tf.nn.avg_pool(inputs, [1, pool_size, pool_size, 1], 
                              strides=[1, pool_size, pool_size, 1], 
                              padding="VALID")

def fc(inputs, n_out, use_bias=True, scope='fc'):
    inputs_shape = inputs.get_shape().as_list()
    n_in = inputs_shape[-1]
    with tf.variable_scope(scope):
        weight = create_variable("weight", 
                                 shape=[n_in, n_out], 
                                 initializer=tf.random_normal_initializer(stddev=0.01))
        if use_bias:
            bias = create_variable("bias", 
                                   shape=[n_out,], 
                                   initializer=tf.zeros_initializer())
            return tf.nn.xw_plus_b(inputs, weight, bias)
        return tf.matmul(inputs, weight)

class MobileNetV1(object):
    def __init__(self, inputs, num_classes=1000, is_training=True, scope="MobileNet"):
        self.inputs = inputs
        self.num_classes = num_classes
        self.is_training = is_training
        
        with tf.variable_scope(scope):
            with tf.variable_scope('conv_1'):
                net = conv2d(inputs, 32, filter_size=3, strides=2)
                net = bacth_norm(net, is_training=self.is_training)
                net = relu(net)
            net = self._depthwise_separable_conv2d(net, 64, "ds_conv_2")
            net = self._depthwise_separable_conv2d(net, 128, "ds_conv_3", downsample=True)
            net = self._depthwise_separable_conv2d(net, 128, "ds_conv_4")
            net = self._depthwise_separable_conv2d(net, 256, "ds_conv_5", downsample=True)
            net = self._depthwise_separable_conv2d(net, 256, "ds_conv_6")
            net = self._depthwise_separable_conv2d(net, 512, "ds_conv_7", downsample=True)
            net = self._depthwise_separable_conv2d(net, 512, "ds_conv_8")
            net = self._depthwise_separable_conv2d(net, 512, "ds_conv_9")
            net = self._depthwise_separable_conv2d(net, 512, "ds_conv_10")
            net = self._depthwise_separable_conv2d(net, 512, "ds_conv_11")
            net = self._depthwise_separable_conv2d(net, 512, "ds_conv_12")
            net = self._depthwise_separable_conv2d(net, 1024, "ds_conv_13", downsample=True)
            net = self._depthwise_separable_conv2d(net, 1024, "ds_conv_14")
            net = avg_pool(net, 7, scope="avg_pool_15")
            net = tf.squeeze(net, [1, 2], name="SpatialSqueeze")
            self.logits = fc(net, self.num_classes, scope="fc_16")
            self.predictions = tf.nn.softmax(self.logits)

    def _depthwise_separable_conv2d(self, inputs, num_filters, scope, downsample=False):
        strides = 2 if downsample else 1
        with tf.variable_scope(scope):
            net = depthwise_conv2d(inputs, strides=strides, scope="dw_conv")
            net = bacth_norm(net, is_training=self.is_training, scope="dw_bn")
            net = relu(net, scope='dw_relu')
            net = conv2d(net, num_filters, scope="pw_conv")
            net = bacth_norm(net, is_training=self.is_training, scope="pw_bn")
            net = relu(net, scope='pw_relu')
            return net

if __name__ == '__main__':
    None





