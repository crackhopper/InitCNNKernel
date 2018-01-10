import types
import importlib

FUNC_FULL_DISPLAY=False
class Param(object):
  def __init__(self,**kwargs):
    self.__dict__ = kwargs

  def print_str(self):
    info = []
    for n in self.print_names:
      value = self.__dict__[n]
      if isinstance(value,types.FunctionType):
        if FUNC_FULL_DISPLAY:
          value = value.__module__ +'.'+ value.__name__
        else:
          value = value.__name__
      info.append('%s=%s'%(n,str(value)))
    return ','.join(info)

  def __str__(self):
    info=[]
    for k in self.__dict__:
      value = self.__dict__[k]
      if isinstance(value,types.FunctionType):
        if FUNC_FULL_DISPLAY:
          value = value.__module__ +'.'+ value.__name__
        else:
          value = value.__name__
      info.append('%s=%s'%(k,str(value)))
    return ','.join(info)

  def __repr__(self):
    return 'Param(%s)'%self

  def __getitem__(self,key):
    return self.__dict__[key]

  def __iter__(self):
    return iter(self.__dict__)

  def __eq__(self,other):
    return self.__dict__==other.__dict__

  def copy(self):
    obj = Param()
    obj.__dict__ = self.__dict__.copy()
    return obj


# NOTE: all the class that reference the network object should be a component.
class Component(object):
  def __init__(self,netobj,**kwargs):
    assert netobj is not None
    self.net = netobj
    self.param = Param(**kwargs)

  def to_pickle(self):
    o = {}
    o['module'] = type(self).__module__
    o['typename'] = type(self).__name__
    o['param'] = self.param  # all the data need to be saved
    return o

  @staticmethod
  def restore(o,net):
    module = importlib.import_module(o['module'])
    classObj = getattr(module,o['typename'])
    if classObj is None:
      raise RuntimeError("cannot import %s from %s"%(o['typename'],o['module']))
    return classObj(net,**(o['param'].__dict__))


