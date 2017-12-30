# inialize all: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/variables.py#L1139
# how to call initializer: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/variable_scope.py#L686

import tensorflow as tf
import numpy as np
import inspect
from tfs.core.elem import Param,Component

class InitType(object):
  values = 0
  ops = 1

class Initializer(Component):
  def __init__(self,netobj,**kwargs):
    super(Initializer,self).__init__(netobj,**kwargs)
    self._init_table=()

  def __str__(self):
    plist = [
      type(self).__name__,
      '-----param-----',
      str(self.param),
      '-----nodes-----'
    ]
    # get print table
    init_tbl = self.init_table[1]
    print_tbl = {}
    for n,val in init_tbl.items():
      node_name=self.net.variables[n].tfs_node.name
      if node_name not in print_tbl:
        print_tbl[node_name]={}
      print_tbl[node_name][n] = val.for_print

    # format print
    for n in self.net.net_def:
      plist.append(n.name)
      if n.name in print_tbl:
        for k,v in print_tbl[n.name].items():
          plist.append('    %-20s%s'%(k,v))
    return '\n'.join(plist)

  def _debug_print_op(self):
    for k,v in self.init_table[1].items():
      vnet = self.net.variables[k]
      print k,vnet.name, vnet.device,'<-',v.name,v.device

  @property
  def init_table(self):
    if self._init_table is ():
      with self.net.graph.as_default(),tf.device('/cpu:0'):
        self._init_table = (self.ret_type,self._build_init_table())
        for k,v in self._init_table[1].items():
          assert 'CPU' in v.device
    return self._init_table

  def _build_init_table(self):
    tbl = {}
    for n in self.net.net_def:
      for name in n.variables:
        v = n.variables[name]
        self._cur_node = n
        self._cur_vname = name
        if self.ret_type == InitType.values:
          tbl[v.name] = self.init_layer_by_val(v.get_shape().as_list(),v.dtype.base_dtype)
          assert isinstance(tbl[v.name],tf.Tensor) or isinstance(tbl[v.name],np.ndarray)
        elif self.ret_type == InitType.ops:
          tbl[v.name] = self.init_layer_by_op(v)
          assert isinstance(tbl[v.name],tf.Tensor)
        else:
          raise ValueError("Unsupport intializier type: %d"%self.ret_type)
        self._cur_node = None
        self._cur_vname = None
    return tbl

  def init_layer_by_val(self,shape,dtype):
    """implement by subclass and return the initial value
    """
    raise NotImplementedError("Should be implemented by subclass")

  def init_layer_by_op(self,var):
    """implement by subclass and return the initial value
    """
    raise NotImplementedError("Should be implemented by subclass")


  def compute(self):
    t,initor = self.init_table
    if t == InitType.values:
      return self._get_init_op_by_val(initor)
    elif t == InitType.ops:
      return self._get_init_op_by_ops(initor)
    else:
      raise ValueError("not support initor type")

  def op_by_value_table(self,tbl):
    return self._get_init_op_by_val(tbl)

  def _get_init_op_by_val(self,tbl):
    return self._get_init_op_by_ops({
      n:tf.assign(self.net.variables[n],val)
      for n,val in tbl.items()
    })

  def _get_init_op_by_ops(self,initor):
    return tf.group(*initor.values())


# subclass from the following classes:
class DefaultValueInit(Initializer):
  ret_type = InitType.values
  def init_layer_by_val(self,shape,dtype):
    """implement by subclass and return the initial value
    """
    raise NotImplementedError("Should be implemented by subclass")

class DefaultOpInit(Initializer):
  ret_type = InitType.ops
  def init_layer_by_op(self,var):
    """implement by subclass and return the initial value
    """
    raise NotImplementedError("Should be implemented by subclass")

