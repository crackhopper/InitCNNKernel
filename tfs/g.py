# Define the global singletons used in the package
from os.path import expanduser
import json
import os
from tfs.core.util import Singleton
from tfs.core.elem import Param,Component
import atexit

_home = expanduser("~")

############ Configuration of the package
class _GlobalConfig(Singleton):
  _cfgfile = _home+'/.tfs.json'
  def __init__(self):
    if os.path.exists(self._cfgfile):
      f = open(self._cfgfile,'r')
      self._cfg = json.load(f)
      f.close()
    else:
      self._cfg={}

    # default configuration for submodules
    self.dataset = DatasetConfig(self['dataset'])

    # test config
    test_config = self['test']
    test_configurable_keys = ['optimizer'] # the key that can be configured
    for k in test_configurable_keys:
      test_config[k]=test_config.get(k,True)

  def save_to_disk(self):
    f = open(self._cfgfile,'w')
    json.dump(self._cfg,f,indent=4, sort_keys=True)
    f.close()

  def __getitem__(self,key):
    if key in self._cfg:
      return self._cfg[key]
    else:
      self._cfg[key]={}
      return self._cfg[key]

class DatasetConfig(object):
  def __init__(self,cfgobj):
    if 'basedir' not in cfgobj:
      cfgobj['basedir'] = _home+'/tfs_data/'
    if 'loaded' not in cfgobj:
      cfgobj['loaded'] = {}
    if 'downlaod' not in cfgobj:
      cfgobj['download'] = _home+'/tfs_data/'+"download"
    self.cfg = cfgobj

  def getdir(self,dataset):
    typename = type(dataset).__name__
    if typename in self.cfg['loaded']:
      return self.cfg['loaded'][typename]
    return None

  def setdir(self,dataset,_dir):
    typename = type(dataset).__name__
    self.cfg['loaded'][typename] = _dir

  @property
  def base_dir(self):
    return self.cfg['basedir']

  @property
  def download_dir(self):
    return self.cfg['download']

config = _GlobalConfig()
atexit.register(config.save_to_disk)
############ Configuration of the package END

############ Name Manager
class _NameManager(Singleton):
  def __init__(self):
    # mapping the network object and its name set
    self.used_names = {}
    self.name_counters={}

  def get_unique_name(self,obj):
    assert isinstance(obj,Component)
    objnet = obj.net
    if objnet not in self.used_names:
      self.used_names[objnet] = set()
      self.name_counters[objnet] = 0

    nameset = self.used_names[objnet]
    self.name_counters[objnet]+=1

    name = getattr(obj.param,'name',None)
    if name is None:
      return '%s_%d'%(type(obj).__name__,self.name_counters[objnet])

    assert isinstance(name,str)
    if name in nameset:
      return '%s_%d'%(name,self.name_counters[objnet])
    else:
      return name

name_manager = _NameManager()
############ Name Manager END
