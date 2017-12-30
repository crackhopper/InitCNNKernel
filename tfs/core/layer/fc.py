import tensorflow as tf
import numpy as np
import ops
from base import Layer
import tfs.core.initializer.init_func as init

class FullyConnect(Layer):
  def __init__(self,
               net,
               outdim,
               activation = ops.relu,
               name=None,
               print_names=['outdim','activation']
  ):
    vtable = locals()
    del vtable['self']
    del vtable['net']
    super(FullyConnect,self).__init__(net,**vtable)

  def _build(self):
    inTensor = self._in
    input_shape = inTensor.get_shape()
    if input_shape.ndims == 4:
      # The input is spatial. Vectorize it first.
      dim = np.prod(input_shape.as_list()[1:])
      output = tf.reshape(inTensor, [-1,dim])
    else:
      output, dim = (inTensor, input_shape[-1].value)
    weights = self._make_variable('weights', shape=[dim, self.param.outdim],init=init.xavier())
    biases = self._make_variable('biases', [self.param.outdim],init=init.constant())
    output = tf.nn.xw_plus_b(output, weights, biases,name=self.name)
    if self.param.activation:
      output= self.param.activation(output, name=self.name)
    return output

  def _inverse(self):
    outTensor = self._inv_in
    name = 'inv_'+self.name
    act = self.param.activation
    if act:
      outTensor = act(outTensor)
    weights = tf.transpose(self._variables['weights'])
    inv_fc = tf.matmul(outTensor,weights)
    shape = self._in.get_shape().as_list()
    shape[0]=-1
    inv_fc = tf.reshape(inv_fc,shape)
    print 'inv_fc '+str(outTensor.get_shape().as_list()) + '->' + str(inv_fc.get_shape().as_list())
    return inv_fc

