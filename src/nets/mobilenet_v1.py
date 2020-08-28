import tensorflow as tf

def create_variable(name, shape, initializer, dtype=tf.float32, trainable=True):
    return tf.get_variable(name, shape=shape, dtype=dtype, 
                           initializer=initializer, 
                           trainable=trainable)
    
def conv2d(inputs, scope, num_filters, filter_size=1, strides=1):
    inputs_shape = inputs.get_shape().as_list()
    in_channels = inputs_shape[-1]
    with tf.variable_scope(scope):
        weights = create_variable("Filter", 
                                  shape=[filter_size, filter_size, in_channels, num_filters], 
                                  initializer=tf.truncated_normal_initializer(stddev=0.01))
        return tf.nn.conv2d(inputs, weights, strides=[1, strides, strides, 1], padding="SAME")

def depthwise_conv2d(inputs, scope, filter_size=3, channel_multiplier=1, strides=1):
    inputs_shape = inputs.get_shape().as_list()
    in_channels = inputs_shape[-1]
    with tf.variable_scope(scope):
        weights = create_variable("Filter", 
                                  shape=[filter_size, filter_size, in_channels, channel_multiplier], 
                                  initializer=tf.truncated_normal_initializer(stddev=0.01))
        return tf.nn.depthwise_conv2d(inputs, weights, strides=[1, strides, strides, 1], 
                                      padding="SAME", rate=[1, 1])

def bacth_norm(inputs, scope, epsilon=1e-05, momentum=0.99, is_training=True):
    with tf.variable_scope(scope):
        return tf.layers.batch_normalization(inputs, 
                                             momentum=momentum, 
                                             epsilon=epsilon, 
                                             training=is_training)

def avg_pool(inputs, pool_size, scope):
    with tf.variable_scope(scope):
        return tf.nn.avg_pool(inputs, [1, pool_size, pool_size, 1], 
                              strides=[1, pool_size, pool_size, 1], 
                              padding="VALID")

def fc(inputs, n_out, scope, use_bias=True):
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
    def __init__(self, 
                 inputs, 
                 num_classes=1000, 
                 is_training=True,
                 width_multiplier=1, 
                 scope="MobileNet"):
        self.inputs = inputs
        self.num_classes = num_classes
        self.is_training = is_training
        self.width_multiplier = width_multiplier

        with tf.variable_scope(scope):
            net = conv2d(inputs, "conv_1", round(32 * width_multiplier), filter_size=3,
                         strides=2)
            net = tf.nn.relu(bacth_norm(net, "conv_1/bn", is_training=self.is_training))
            net = self._depthwise_separable_conv2d(net, 64, self.width_multiplier,
                                "ds_conv_2")
            net = self._depthwise_separable_conv2d(net, 128, self.width_multiplier,
                                "ds_conv_3", downsample=True)
            net = self._depthwise_separable_conv2d(net, 128, self.width_multiplier,
                                "ds_conv_4")
            net = self._depthwise_separable_conv2d(net, 256, self.width_multiplier,
                                "ds_conv_5", downsample=True)
            net = self._depthwise_separable_conv2d(net, 256, self.width_multiplier,
                                "ds_conv_6")
            net = self._depthwise_separable_conv2d(net, 512, self.width_multiplier,
                                "ds_conv_7", downsample=True)
            net = self._depthwise_separable_conv2d(net, 512, self.width_multiplier,
                                "ds_conv_8")
            net = self._depthwise_separable_conv2d(net, 512, self.width_multiplier,
                                "ds_conv_9")
            net = self._depthwise_separable_conv2d(net, 512, self.width_multiplier,
                                "ds_conv_10")
            net = self._depthwise_separable_conv2d(net, 512, self.width_multiplier,
                                "ds_conv_11")
            net = self._depthwise_separable_conv2d(net, 512, self.width_multiplier,
                                "ds_conv_12")
            net = self._depthwise_separable_conv2d(net, 1024, self.width_multiplier,
                                "ds_conv_13", downsample=True)
            net = self._depthwise_separable_conv2d(net, 1024, self.width_multiplier,
                                "ds_conv_14")
            net = avg_pool(net, 7, "avg_pool_15")
            net = tf.squeeze(net, [1, 2], name="SpatialSqueeze")
            self.logits = fc(net, self.num_classes, "fc_16")
            self.predictions = tf.nn.softmax(self.logits)

    def _depthwise_separable_conv2d(self, inputs, num_filters, width_multiplier,
                                    scope, downsample=False):
        num_filters = round(num_filters * width_multiplier)
        strides = 2 if downsample else 1
        with tf.variable_scope(scope):
            net = depthwise_conv2d(inputs, "dw_conv", strides=strides)
            net = bacth_norm(net, "dw_bn", is_training=self.is_training)
            net = tf.nn.relu(net)
            net = conv2d(net, "pw_conv", num_filters)
            net = bacth_norm(net, "pw_bn", is_training=self.is_training)
            return tf.nn.relu(net)


































































