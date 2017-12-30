import tensorflow as tf
import numpy as np
import ops
from base import Layer

class MaxPool(Layer):
  def __init__(self,
               net,
               ksize,
               strides,
               padding='SAME',
               name=None,
               print_names=['ksize','strides']
  ):
    vtable = locals()
    del vtable['self']
    del vtable['net']
    super(MaxPool,self).__init__(net,**vtable)

  def _build(self):
    inTensor = self._in
    kx,ky = self.param.ksize
    sx,sy = self.param.strides
    if (self.net.data_format=='NHWC'):
      ksize=[1,kx,ky,1]
      strides=[1,sx,sy,1]
    else:
      ksize=[1,1,kx,ky]
      strides=[1,1,sx,sy]

    output = tf.nn.max_pool(
      inTensor,
      ksize=ksize,
      strides=strides,
      padding=self.param.padding,
      data_format=self.net.data_format,
      name=self.name)
    return output

  def _inverse(self):
    outTensor = self._inv_in
    name = 'inv_'+self.name
    outshape = self._out.get_shape().as_list()
    outshape[0]=-1 # the first dimension is used for batch
    if outTensor.get_shape().ndims != 4:
      outTensor = tf.reshape(outTensor,outshape)
    print outTensor.get_shape()
    out = ops.max_unpool(outTensor,self._out,name)
    print 'inv_max_pool ' + str(outTensor.get_shape().as_list()) + '->' + str(out.get_shape().as_list())
    self._inv_out = out
    return out


class AvgPool(Layer):
  def __init__(self,
               net,
               ksize,
               strides,
               padding='SAME',
               name=None,
               print_names=['ksize','strides']
  ):
    vtable = locals()
    del vtable['self']
    del vtable['net']
    super(MaxPool,self).__init__(net,**vtable)

  def build(self,inTensor):
    self._in = inTensor
    kx,ky = self.param.ksize
    sx,sy = self.param.strides
    if (self.net.data_format=='NHWC'):
      ksize=[1,kx,ky,1]
      strides=[1,sx,sy,1]
    else:
      ksize=[1,1,kx,ky]
      strides=[1,1,sx,sy]
    with tf.variable_scope(self.name) as scope:
      output = tf.nn.avg_pool(inTensor,
                              ksize=ksize,
                              strides=strides,
                              padding=self.param.padding,
                              data_format=self.net.data_format,
                              name=self.name)
    self._out = output
    return output
