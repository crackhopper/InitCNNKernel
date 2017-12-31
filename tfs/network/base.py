import numpy as np
from tfs.core.util import run_once_for_each_obj
from tfs.core.initializer import DefaultInit
from tfs.core.loss import DefaultLoss
from tfs.core.regularizers import DefaultRegularizer
from tfs.core.monitor import DefaultMonitor
from tfs.core.optimizer import DefaultOptimizer
from tfs.core.layer import func_table,Layer
from tfs.core.elem import Component
from tfs.core.learning_rate import DefaultLearningRate
import pickle
import new

import tensorflow as tf
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.client import device_lib
from sklearn import metrics

# for supporting multi-gpu:
# https://github.com/tensorflow/tensorflow/blob/r0.7/tensorflow/models/image/cifar10/cifar10_multi_gpu_train.py#L174
#
# we use shared variables on CPU and model distributed on each GPU

from net_struct import NetStructure
#################### Network
# decorators
def with_graph(f):
  def with_graph_run(self,*args,**kwargs):
    with self.graph.as_default():
      return f(self,*args,**kwargs)
  # this is important to make the decorator compatiable with run_once_each_obj.
  with_graph_run.__name__=f.__name__
  return with_graph_run

class Network(object):
  __hash__=object.__hash__
  def __init__(self):
    self._init_graph_sess()
    self._struct = NetStructure(self)

    self._true_out=None
    self._in = None
    self._out = None
    self._loss=None

    self.variables = {}
    self.initializer = DefaultInit(self)
    self.losser = DefaultLoss(self)
    self.regularizer =DefaultRegularizer(self)
    self.monitor = {}
    self.monitor['default']=DefaultMonitor(self)
    # learning rate must precede the construction of optimizer
    self.lr = DefaultLearningRate(self,0.001)
    self._optimizer = DefaultOptimizer(self)

    # this must be set when define a network
    self.loss_input_layer_name = None

    self._regulization=None
    self.grads = None
    self._train_op = None

    self.num_gpu = 0
    self.i_step = 0
    self.n_epoch = 0
    self._dtype = None

  def to_pickle(self):
    return [
      self.in_shape,
      self.loss_input_layer_name,
      self.lr.to_pickle(),
      self.optimizer.to_pickle(),
      self.losser.to_pickle(),
      self.regularizer.to_pickle(),
      self.data_format
    ]

  def restore(self,objs):
    inshape = objs[0]
    self.loss_input_layer_name = objs[1]
    self.lr = Component.restore(objs[2],self)
    self.optimizer = Component.restore(objs[3],self)
    self.losser = Component.restore(objs[4],self)
    self.regularizer = Component.restore(objs[5],self)
    self.data_format = objs[6]

    if inshape:
      self.build(inshape)
    

  def _init_graph_sess(self):
    self._graph = tf.Graph()
    with self.graph.as_default():
      self._sess = tf.Session()

  @property
  def optimizer(self):
    return self._optimizer

  @optimizer.setter
  def optimizer(self,opt):
    self.grads=None
    self._optimizer=opt

  def add_monitor(self,name,monitor):
    self.monitor[name] = monitor

  @staticmethod
  def available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x for x in local_device_protos]

  def __len__(self):
    return len(self.net_def)

  @property
  @deprecated("2017-05-01", "Use `net_def` instead.")
  def layers(self):
    return self._struct

  @property
  def nodes(self):
    return self._struct

  @property
  def net_def(self):
    return self._struct

  def node_to_index(self,l):
    return self.net_def.find_index(l)

  def node_by_index(self,idx):
    return self.net_def[idx]

  @deprecated("2017-05-01", "Use `node_by_name` instead.")
  def layer_by_name(self,name):
    return self.net_def.by_name(name)

  def node_by_name(self,name):
    return self.net_def.by_name(name)

  def __del__(self):
    self.sess.close()

  def setup(self):
    '''Construct the network. '''
    raise NotImplementedError('Must be implemented by the subclass.')

  def setup_with_def(self,struct_def,in_shape=None):
    if isinstance(struct_def,list):
      struct_def = NetStructure(self,nodes=struct_def)
    self._struct = struct_def.copy_to(self)
    if in_shape:
      self.build(in_shape)

  @property
  def graph(self):
    return self._graph

  @property
  def input(self):
    return self._in

  @property
  def output(self):
    return self._out

  @property
  def true_output(self):
    return self._true_out

  @property
  def sess(self):
    return self._sess

  def _init_in_out_size(self):
    if self.num_gpu and self._in is None and self._out is None:
      self._in = [None]*self.num_gpu
      self._out = [None]*self.num_gpu
      self._true_out = [None]*self.num_gpu
      self._loss = [None]*self.num_gpu

  def tf_graph_str(self):
    info=[]
    for n in self.graph.as_graph_def().node:
      s = '%-20s@%20s'%(n.name,n.device)
      if hasattr(n,'tfs_nodename'):
        s=s+' --%s'%n.tfs_nodename
      info.append(s)
    return '\n'.join(info)

  @with_graph
  @run_once_for_each_obj
  def build(self,input_shape,dtype=tf.float32):
    self._dtype = dtype
    """Build the computational graph
    inTensor: the network input tensor.
    """
    if not self.num_gpu:
      self._build(input_shape,dtype)
    else:
      tower_grads = []
      for i in range(self.num_gpu):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % ('GPU', i)) as scope:
            self._build(input_shape,dtype,i)
            tf.get_variable_scope().reuse_variables()
            _loss = self.loss[i]
            tower_grads.append(_grad)

    self.build_variables_table()
    self._initialize()
    self.compute_gradients()
    return self.output

  def compute_gradients(self):
    if self.loss is None:
      return
    if not self.num_gpu:
      self.grads = self.optimizer.compute_gradients(self.loss,self.variables.values())
    else:
      tower_grads = []
      for i in range(self.num_gpu):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % ('GPU', i)) as scope:
            tf.get_variable_scope().reuse_variables()
            _loss = self.loss[i]
            _grad = self.optimizer.compute_gradients(_loss,self.variables.values())
            tower_grads.append(_grad)
      self.grads = self.average_gradients(tower_grads)

  def average_gradients(self,tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
      # Note that each grad_and_vars looks like the following:
      #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
      grads = []
      for g, _ in grad_and_vars:
        expanded_g = tf.expand_dims(g, 0)
        grads.append(expanded_g)
      grad = tf.concat(axis=0, values=grads)
      grad = tf.reduce_mean(grad, 0)
      v = grad_and_vars[0][1]
      grad_and_var = (grad, v)
      average_grads.append(grad_and_var)
    return average_grads

  # this function is called only in build() under current graph.
  def _build(self,input_shape,dtype,idx=None):
    self._init_in_out_size()
    tmp = tf.placeholder(dtype,input_shape)
    if idx is None:
      self._in = tmp
    else:
      self._in[idx] = tmp

    for l in self.net_def:
      tmp = l.build(tmp,idx)

    if idx is None:
      self._out = tmp
      output_shape=self._out.get_shape().as_list()
      output_dtype=self._out.dtype
      self._true_out=tf.placeholder(dtype=output_dtype,shape=output_shape)
      self._loss = self._compute_loss(idx)
    else:
      self._out[idx] = tmp
      output_shape=self._out[idx].get_shape().as_list()
      output_dtype=self._out[idx].dtype
      self._true_out[i]=tf.placeholder(dtype=output_dtype,shape=output_shape)
      self._loss[idx] = self._compute_loss(idx)
    return self

  def _initialize(self):
    self.run_initor(self.initializer)

  def _compute_loss(self,idx):
    loss =  self.losser.compute(idx)
    if loss is None:
      return loss
    return loss + self.regularizer.compute()

  @property
  def loss(self):
    return self._loss

  def build_variables_table(self):
    for l in self.net_def:
      for k in l.variables:
        v = l.variables[k]
        self.variables[v.name] = v

  def has_built(self):
    if hasattr(self,'_has_run'):
      if Network.build.__name__ in self._has_run:
        return True
    return False

  def fit(self,dataset,batch_size,n_epoch,
          shuffle_epoch=True,max_step=10000000):
    if dataset.train.labels.shape[-1] != self.out_shape[-1]:
      dataset = dataset.to_one_hot()
    train_set = dataset.train
    test_set = dataset.test
    train_set.before_iter()
    self.i_step = self.run(self.lr.step)
    self.n_epoch = 0
    while True:
      self.n_epoch = train_set.epochs_completed
      X,y = train_set.next_batch(batch_size,shuffle=shuffle_epoch)
      self.step(X,y,self.i_step)
      for v in self.monitor.values():
        v.status(train_set,test_set,self.i_step,self.n_epoch)
      if self.n_epoch>=n_epoch:
        break
      if self.i_step >= max_step:
        break
      self.i_step += 1
    return self

  @property
  def train_op(self):
    if self._train_op is None:
      self._train_op = self._get_train_op()
    return self._train_op

  @with_graph
  def _get_train_op(self):
    if self.loss is None:
      return None
    if self.grads is None:
      self.compute_gradients()
    op = self.optimizer.apply_gradients(self.grads)
    # initialize the uninitalized variable (the optimizer would introduce
    # uninitalized variable)
    vars = self.optimizer.variables
    self.run(tf.variables_initializer(vars.values()))
    return op

  def step(self,X,y,step):
    stepop=self.lr.step.assign(step)
    self.run(stepop)
    self.run(self.train_op,feed_dict={self.input:X,self.true_output:y})

  def predict(self,X):
    if self.num_gpu==0:
      _in = self.input
      _out = self.output
    else:
      _in = self.input[0]
      _out = self.output[0]
    return self.run(_out,feed_dict={_in:X})

  def eval_node_input(self,node,X):
    _in = self.input
    if isinstance(node,str):
      _out = self.node_by_name(node).input
    else:
      _out = node.input
    return self.run(_out,feed_dict={_in:X})

  def eval_node(self,node,X):
    _in = self.input
    if isinstance(node,str):
      _out = self.node_by_name(node).output
    else:
      _out = node.output
    return self.run(_out,feed_dict={_in:X})

  def function(self,input_tensors,output_tensors):
    def _func(input_vals):
      feed = {t:v in zip(input_vals,input_tensors)}
      return self.run(output_tensors,feed_dict=feed)
    return _func

  def score(self,datasubset):
    y_pred = self.predict(datasubset.data)
    y_pred = np.argmax(y_pred,1)
    y_true = datasubset.labels
    y_true = np.argmax(y_true,1)
    return metrics.accuracy_score(y_true,y_pred)

  def measure_loss(self,X,y):
    if self.num_gpu==0:
      _in = self.input
      _true_out = self.true_output
      _loss = self.loss
    else:
      _in = self.input[0]
      _true_out = self.true_output[0]
      _loss = self.loss[0]

    return self.run(_loss,feed_dict={_in:X,_true_out:y})

  def run(self,eval_list,feed_dict=None):
    return self.sess.run(eval_list, feed_dict=feed_dict)

  def run_initor(self,initor):
    op = initor.compute()
    return self.sess.run(op)

  def save(self,filename):
    self.save_def(filename)
    to_save={}
    for k,v in self.variables.items():
      to_save[k]=self.run(v)
    f=open(filename+'.model','wb')
    pickle.dump(to_save,f)
    f.close()

  def save_def(self,filename):
    self.net_def.save(filename+'.modeldef')

  def load(self,filename):
    self._init_graph_sess()
    self.load_def(filename)
    f=open(filename+'.model','rb')
    data_dict=pickle.load(f)
    f.close()
    if self.has_built():
      with self._graph.as_default():
        op = self.initializer.op_by_value_table(data_dict)
      self.run(op)

  def load_def(self,filename):
    self.net_def.load(filename+'.modeldef')

  @property
  def in_shape(self):
    if self._in is not None:
      if self.num_gpu==0:
        return self._in.get_shape().as_list()
      else:
        return self._in[0].get_shape().as_list()
    return None

  @property
  def dtype(self):
    return self._dtype

  @property
  def out_shape(self):
    if self._out is not None:
      if self.num_gpu==0:
        return self._out.get_shape().as_list()
      else:
        return self._out[0].get_shape().as_list()
    return None

  def copy(self):
    obj = Network()
    obj.loss_input_layer_name = self.loss_input_layer_name
    obj.setup_with_def(self.net_def,self.in_shape)
    return obj

  def __str__(self):
    return '\n'.join([str(l) for l in self.nodes])

  def print_shape(self):
    for l in self.nodes:
      print '%-20s  %20s %s %-20s'%(
        l.name,
        l.input.get_shape(),
        '->',
        l.output.get_shape())

  def subnet(self,begin_index,end_index):
    obj = Network()
    obj.setup_with_def(self.layers[begin_index:end_index])
    return obj


class CustomNetwork(Network):
  """Automatically called setup and build when construct
  """
  def __init__(self):
    Network.__init__(self)
    self.default_in_shape = None
    self.setup()
    in_shape = self.default_in_shape
    if hasattr(self,'default_data_format'):
      self.data_format = self.default_data_format
    else:
      self.data_format = 'NHWC'
    if not in_shape:
      raise ValueError("must sepecify the default_in_shape attributes, or pass the shape as an argument when construction")

  def setup(self):
    raise NotImplementedError("CustomNetwork Must Implement setup Method")

  def build(self,inshape=None):
    inshape = inshape or self.default_in_shape
    return Network.build(self,inshape)


