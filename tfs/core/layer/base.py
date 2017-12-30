import numpy as np
import inspect
import tensorflow as tf

import tfs.g
from tfs.core.elem import Param,Component

def local_variable_scope(f):
  """
  """
  def wrapper(self,*args, **kwargs):
    with tf.variable_scope(self.name):
      return f(self,*args, **kwargs)
  wrapper.__name__ = f.__name__
  return wrapper

class Layer(Component):
  def __init__(self,netobj,**kwargs):
    super(Layer,self).__init__(netobj,**kwargs)
    self.maybe_rename()
    # internal used properties
    self._in = None
    self._out = None
    self._variables = {}
    self._initializers = {}

  def maybe_rename(self):
    self.param.name = tfs.g.name_manager.get_unique_name(self)

  @property
  def name(self):
    return self.param.name

  @property
  def variables(self):
    return self._variables

  @property
  def initializers(self):
    return self._initializers

  @property
  def input(self):
    if self.num_gpu:
      return self._inlist
    else:
      return self._in

  @property
  def output(self):
    if self.num_gpu:
      return self._outlist
    else:
      return self._out

  @property
  def num_gpu(self):
    if self.net:
      return self.net.num_gpu
    return 0

  def _init_in_out_size(self):
    if self.num_gpu:
      self._inlist = [None]*self.num_gpu
      self._outlist = [None]*self.num_gpu

  def build(self,inTensor,idx=None):
    self._init_in_out_size()
    self._in = inTensor
    if self.num_gpu:
      self._inlist[idx] = inTensor
      self._outlist[idx] = self._build()
      self._out = self._outlist[idx]
    else:
      self._out = self._build()
    return self._out

  @local_variable_scope
  def _make_variable(self,vname,shape,init):
    # variables should be created on CPU
    with tf.device('/cpu:0'):
      v=tf.get_variable(vname, shape=shape)
      v.tfs_node = self
      v.tfs_basename = vname
      self._variables[vname]=v
      self._initializers[vname]=init
      return v

  def set_weights(self,weight_table):
    with self.net.graph.as_default():
      assign_table = {
        n:tf.assign(self.variables[n],val)
        for n,val in weight_table.items()
      }
      op = tf.group(*assign_table.values())
      self.net.run(op)

  def _build(self):
    '''Run the layer. '''
    raise NotImplementedError('Must be implemented by the subclass.')

  def inverse(self,outTensor):
    self._inv_in = outTensor
    self._inv_out = self._inverse()
    return self._inv_out

  def _inverse(self):
    print '%s doesn\'t define inverse op, ignore the layer'% type(self).__name__
    return self._inv_in

  def copy_to(self,to_net):
    cls = type(self)
    obj = cls(to_net,**self.param.__dict__)
    return obj

  def __str__(self):
    """
    The info of this layer
    """
    return 'Name:%-10s\tType:%s(%s)'%(self.name,type(self).__name__,self.param.print_str())
