import tensorflow as tf
from tfs.core.elem import Param,Component

class LearningRate(Component):
  def __init__(self,netobj,init_value,step,**kwargs):
    self._init_val = init_value
    super(LearningRate,self).__init__(netobj,init_value=init_value,step=step,**kwargs)
    with self.net.graph.as_default():
      self.global_step = tf.Variable(step,dtype=tf.int32, trainable=False)
      self.net.run(self.global_step.initializer)

  @property
  def step(self):
    return self.global_step

  def to_pickle(self):
    with self.net.graph.as_default():
      self.param.step = self.net.run(self.global_step)
    return super(LearningRate,self).to_pickle()

  @property
  def init_value(self):
    return self._init_val

  @property
  def value(self):
    with self.net.graph.as_default():
      return self.net.run(self.variable)

  @property
  def variable(self):
    raise NotImplementedError('Must be implemented by the subclass.')

  def __str__(self):
    return '%s(%s)'%(type(self).__name__,self.param)

class Constant_LR(LearningRate):
  def __init__(self,netobj,init_value,step=0):
    super(Constant_LR,self).__init__(netobj,init_value,step)
    with self.net.graph.as_default():
      self._var = tf.Variable(init_value, trainable=False)
      self.net.run(self._var.initializer)

  @property
  def variable(self):
    return self._var


class ExponentialDecay_LR(LearningRate):
  '''
  see https://www.tensorflow.org/versions/r0.12/api_docs/python/train/decaying_the_learning_rate
  '''
  def __init__(self,netobj,init_value,step=0,decay_steps=10000,decay_rate=0.96):
    super(ExponentialDecay_LR,self).__init__(
      netobj,init_value,step,decay_steps=decay_steps,decay_rate=0.96)

    self._var = tf.train.exponential_decay(
      self.init_value, self.global_step,
      decay_steps, decay_rate, staircase=True)

  @property
  def variable(self):
    return self._var

DefaultLearningRate = ExponentialDecay_LR
