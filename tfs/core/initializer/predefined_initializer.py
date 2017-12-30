from base import DefaultValueInit, DefaultOpInit
import init_func
import tensorflow as tf
import numpy as np

class DefaultInitializer(DefaultValueInit):
  def __init__(
      self,
      net,
      print_names=[]
  ):
    vs = locals()
    net = vs['net']
    del vs['self']
    del vs['net']
    super(DefaultInitializer,self).__init__(net,**vs)

  def init_layer_by_val(self,shape,dtype):
    layerDefaultOp = self._cur_node.initializers[self._cur_vname]
    return layerDefaultOp(shape,dtype)

class AllConstantInitializer(DefaultValueInit):
  def __init__(
      self,
      net,
      val=0.1,
      print_names=[]
  ):
    vs = locals()
    net = vs['net']
    del vs['self']
    del vs['net']
    super(AllConstantInitializer,self).__init__(net,**vs)

  def init_layer_by_val(self,shape,dtype):
    op = init_func.constant(self.param.val)
    return op(shape,dtype)


class CaffeTensorflowLoader(DefaultOpInit):
  def __init__(
      self,
      net,
      filename,
      print_names=['filename']
  ):
    vs = locals()
    net = vs['net']
    del vs['self']
    del vs['net']
    super(CaffeTensorflowLoader,self).__init__(net,**vs)
    self.data_dict = np.load(self.param.filename).item()

  def init_layer_by_op(self,var):
    data_dict = self.data_dict
    node = var.tfs_node
    name = var.tfs_basename
    return tf.assign(node.variables[name],data_dict[node.name][name])

