import tensorflow as tf
import numpy as np
import ops
from base import Layer

class Softmax(Layer):
  def __init__(self,
               net,
               name=None,
               print_names=[]
  ):
    vtable = locals()
    del vtable['self']
    del vtable['net']
    super(Softmax,self).__init__(net,**vtable)

  def _build(self):
    inTensor = self._in
    output = tf.nn.softmax(inTensor,name=self.name)
    return output



