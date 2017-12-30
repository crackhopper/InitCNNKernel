#
# author:crackhopper
#
# we need unpooling operation to do the network visualization task. like
# described in the paper
# https://scholar.google.com/scholar?q=Zeiler+Visualizing+and+understanding+convolutional+networks
#
# By the hint given in
# https://github.com/tensorflow/tensorflow/issues/2169, we can utilize the
# gradient operation for max_unpooling pass. (just like in deep_vis). So we would
# try to do things in this way.
#
# more hidden ops: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/hidden_ops.txt

from tensorflow.python.ops import gen_nn_ops
import tensorflow as tf
def max_unpool(indata, origin_pooled,name):
    with tf.name_scope(name):
        op = origin_pooled.op
        return gen_nn_ops._max_pool_grad(
            op.inputs[0],
            op.outputs[0],
            indata,
            op.get_attr("ksize"),
            op.get_attr("strides"),
            padding=op.get_attr("padding"),
            data_format=op.get_attr("data_format"))

