import tensorflow as tf
import numpy as np
import ops
from base import Layer

class Dropout(Layer):
  def __init__(self,
               net,
               keep_prob,
               name=None,
               print_names=['keep_prob']
  ):
    vtable = locals()
    del vtable['self']
    del vtable['net']
    super(Dropout,self).__init__(net,**vtable)

  def _build(self):
    inTensor = self._in
    output = tf.nn.dropout(inTensor, self.param.keep_prob,
                           name=self.param.name)
    return output

