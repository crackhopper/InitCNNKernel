import tensorflow as tf
import numpy as np
import inspect
from tfs.core.elem import Param,Component
from tfs.core.learning_rate import DefaultLearningRate

class Optimizer(Component):
  def __init__(self,net,**kwargs):
    super(Optimizer,self).__init__(net,**kwargs)
    self._variables = None

  def __str__(self):
    plist = [
      type(self).__name__,
      '-----param-----',
      str(self.param),
      '---------------'
    ]
    return '\n'.join(plist)

  def compute_gradients(self, loss, var_list,
                        gate_gradients=tf.train.Optimizer.GATE_OP,
                        aggregation_method=None,
                        colocate_gradients_with_ops=False,
                        grad_loss=None):
    return self.opt.compute_gradients(loss, var_list,
                        gate_gradients,
                        aggregation_method,
                        colocate_gradients_with_ops,
                        grad_loss)

  def apply_gradients(self,grads_and_vars,global_step=None,name=None):
    op = self.opt.apply_gradients(grads_and_vars,global_step,name)
    self._init_variable_table(grads_and_vars)
    return op

  def _init_variable_table(self,grads_and_vars):
    if self._variables is None:
      varlist = [self.opt.get_slot(v, name)
                 for g, v in grads_and_vars if g is not None
                 for name in self.opt.get_slot_names()]
      self._variables={}
      for v in varlist:
        self._variables[v.name]=v

  @property
  def variables(self):
    if self._variables is None:
      raise ValueError("The optimize op isn't built")
    return self._variables

class OptimizerWithLearningRate(Optimizer):
  def __init__(self,net,learning_rate,**kwargs):
    super(OptimizerWithLearningRate,self).__init__(net,learning_rate=learning_rate,**kwargs)

  def apply_gradients(self,grads_and_vars,name=None):
    op = self.opt.apply_gradients(grads_and_vars,self.param.learning_rate.global_step,name)
    self._init_variable_table(grads_and_vars)
    return op

class GradientDecentOptimizer(OptimizerWithLearningRate):
  def __init__(self,net,init_learning_rate=0.001,print_names=['learning_rate'],
               LR_Class=DefaultLearningRate):
    learning_rate=LR_Class(net,init_learning_rate)
    super(GradientDecentOptimizer,self).__init__(net,learning_rate,print_names=['learning_rate'])
    self.opt = tf.train.GradientDescentOptimizer(learning_rate.value)

class AdamOptimizer(OptimizerWithLearningRate):
  def __init__(self,net,init_learning_rate=0.001,print_names=['learning_rate'],
               LR_Class=DefaultLearningRate):
    learning_rate=LR_Class(net,init_learning_rate)
    super(AdamOptimizer,self).__init__(net,learning_rate,print_names=['learning_rate'])
    self.opt = tf.train.AdamOptimizer(learning_rate.value)

  def _init_variable_table(self,grads_and_vars):
    super(AdamOptimizer,self)._init_variable_table(grads_and_vars)
    b1,b2 = self.opt._get_beta_accumulators()
    self._variables[b1.name]=b1
    self._variables[b2.name]=b2

DefaultOptimizer=AdamOptimizer
