from tfs.core.initializer import Initializer,InitType
from tfs.core.layer import *
import numpy as np

def svd_orthonormal(shape):
  if len(shape) < 2:
    raise RuntimeError("Only shapes of length 2 or more are supported.")
  flat_shape = (shape[0], np.prod(shape[1:]))
  a = np.random.standard_normal(flat_shape)
  u, _, v = np.linalg.svd(a, full_matrices=False)
  q = u if u.shape == flat_shape else v
  q = q.reshape(shape)
  return q

# this initializer would also change the weight of current net.
class LSUV(Initializer):
  ret_type = InitType.values
  available_node_type = [Conv2d, FullyConnect]
  def __init__(
      self,
      net,
      batchX,
      print_names=[]
  ):
    vs = locals()
    net = vs['net']
    del vs['self']
    del vs['net']
    super(LSUV,self).__init__(net,**vs)

  def _build_init_table(self):
    tbl={}
    margin = 0.1
    max_iter = 10
    for n in self.net.net_def:
      if type(n) not in self.available_node_type:
        for name in n.variables:
          v = n.variables[name]
          tbl[v.name] = n.initializers[name](v.get_shape().as_list(),v.dtype.base_dtype)
        continue
      print(type(n).__name__)
      my_dict = {}

      name = 'weights'
      v = n.variables[name]
      defaultInitOp = n.initializers[name]
      val = defaultInitOp(v.get_shape().as_list(),v.dtype.base_dtype)
      myval = svd_orthonormal(val.shape)
      my_dict[name] = myval

      name = 'biases'
      v = n.variables[name]
      defaultInitOp = n.initializers[name]
      val = defaultInitOp(v.get_shape().as_list(),v.dtype.base_dtype)
      myval = val
      my_dict[name] = myval

      n.set_weights(my_dict)

      acts1 = self.net.eval_node(n,self.param.batchX)
      var1=np.var(acts1)
      iter1=0
      needed_variance = 1.0
      print(var1)

      while (abs(needed_variance - var1) > margin):
        weights = self.net.run(n.variables['weights'])
        biases = self.net.run(n.variables['biases'])
        weights /= np.sqrt(var1)/np.sqrt(needed_variance)
        w_all_new = {'weights':weights,
                     'biases':biases}
        n.set_weights(w_all_new)
        acts1=self.net.eval_node(n,self.param.batchX)
        var1=np.var(acts1)
        iter1+=1
        print(var1)
        if iter1 > max_iter:
          break

    return tbl

