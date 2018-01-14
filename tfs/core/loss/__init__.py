# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import inspect
from tfs.core.elem import Param,Component
from tfs.core.layer.inference import Softmax


_EPSILON=1e-07

def _to_tensor(x, dtype):
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x

class Loss(Component):
    def __init__(self,netobj,**kwargs):
        super(Loss,self).__init__(netobj,**kwargs)

    @property
    def in_name(self):
        return self.net.loss_input_layer_name

    def compute(self,idx=None):
        in_name = self.in_name
        if len(self.net) is 0:
            return None
        if in_name is None:
            raise KeyError("please define loss_input_layer_name")
        else:
            if in_name not in self.net.net_def.names():
                raise KeyError("Loss input layer (%s) doesnot exist"%in_name)
            x1_node = self.net.node_by_name(in_name)
        if idx is None:
            x1 = x1_node.output
            x2 = self.net.true_output
        else:
            x1 = x1_node.output[idx]
            x2 = self.net.true_output[idx]

        return self._compute(x1,x2)

    def _compute(self,x1,x2):
        raise NotImplementedError

    def __str__(self):
        info = type(self).__name__+' (%s)'%self.in_name
        pstr = str(self.param)
        return info+'\n-----param-----\n'+pstr+'----------------'


class CrossEntropyByLogitLabel(Loss):
    def __init__(self,netobj,print_names=[]):
        Loss.__init__(self,netobj)

    def _compute(self,x1,x2):
        num_out = x1.get_shape().as_list()[-1]
        op = None
        if num_out==1:
            op = tf.nn.sigmoid_cross_entropy_with_logits
        else:
            op = tf.nn.softmax_cross_entropy_with_logits
        return tf.reduce_mean(op(labels=x2,logits=x1))

class SquareError(Loss):
    def __init__(self,netobj,print_names=[]):
        Loss.__init__(self,netobj)
    def _compute(self,x1,x2):
        x = tf.reshape(x2-x1,[-1])
        norm2=tf.norm(x,ord='euclidean')
        return tf.square(norm2)

DefaultLoss=CrossEntropyByLogitLabel
