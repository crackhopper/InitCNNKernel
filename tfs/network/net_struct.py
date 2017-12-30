import pickle
from tfs.core.layer import func_table,Layer
from tfs.core.elem import Component
#################### NetStructure
def _layer_function(layerclass):
  def func(self,*args,**kwargs):
    layer = layerclass(self.net,*args,**kwargs)
    self.append(layer)
    return self
  return func

def _net_sturcture_meta(future_class_name, future_class_parents, future_class_attr):
  for k in func_table:
    future_class_attr[k]=_layer_function(func_table[k])
  return type(future_class_name, future_class_parents, future_class_attr)

class NetStructure(object):
  __metaclass__ = _net_sturcture_meta
  """This class is used for define a network structure by using layers.
  """
  def __init__(self,net,nodes=None):
    nodes = nodes or []
    self._net = net
    self.nodes=nodes

  @property
  def in_shape(self):
    return self.net.in_shape

  def _adjust(self):
    def _setnet(n):
      n.net=self.net
    map(_setnet, self._nodes)

  @property
  def net(self):
    return self._net

  @net.setter
  def net(self,net):
    self._net = net
    self._adjust()

  @property
  def nodes(self):
    return self._nodes

  @nodes.setter
  def nodes(self,nodes):
    self._nodes = nodes
    self._need_built = True
    self._adjust()

  def append(self,l):
    self._need_built = True
    self.nodes.append(l)

  def __getitem__(self,i):
    return self.nodes[i]

  def copy_to(self,net):
    res = NetStructure(net)
    for l in self.nodes:
      res.nodes.append(l.copy_to(net))
    return res

  def _built_lut(self):
    if not self._need_built: return
    self._lut = {}
    self._lut2 = {}
    for i,l in enumerate(self.nodes):
      self._lut[l.name]=l
      self._lut2[l.name]=i

  def by_name(self,name):
    self._built_lut()
    return self._lut[name]

  def find_index(self,l):
    self._built_lut()
    return self._lut2[l.name]

  def names(self):
    self._built_lut()
    return self._lut.keys()

  def __len__(self):
    return len(self.nodes)

  def save(self,filename):
    f=open(filename,'wb')
    nodes = [n.to_pickle() for n in self.nodes]
    pickle.dump([nodes,self.net.to_pickle()],f)
    f.close()

  def load(self,filename):
    f=open(filename,'rb')
    nodes,others = pickle.load(f)
    self.nodes = [Component.restore(n,self.net) for n in nodes]
    self.net.restore(others)
    f.close()
