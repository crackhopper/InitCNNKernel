import tensorflow as tf
from tfs.core.elem import Param,Component

class LearningRate(Component):
  def __init__(self,netobj,init_value,**kwargs):
    self._value = init_value
    super(LearningRate,self).__init__(netobj,init_value=init_value,**kwargs)
    with self.net.graph.as_default():
      self.global_step = tf.Variable(0, trainable=False)
      self.net.run(self.global_step.initializer)

  @property
  def init_value(self):
    return self._value

  @property
  def value(self):
    raise NotImplementedError('Must be implemented by the subclass.')

  def __str__(self):
    return '%s(%s)'%(type(self).__name__,self.param)

  @property
  def state(self):
    if(isinstance(self.value,float)):
      return 'lr: %f'%self.value
    else:
      with self.net.graph.as_default():
        return 'lr: %f'%self.net.run(self.value)

class Constant_LR(LearningRate):
  @property
  def value(self):
    return self.init_value


class ExponentialDecay_LR(LearningRate):
  '''
  see https://www.tensorflow.org/versions/r0.12/api_docs/python/train/decaying_the_learning_rate
  '''
  def __init__(self,netobj,init_value,decay_steps=10000,decay_rate=0.96):
    super(ExponentialDecay_LR,self).__init__(netobj,init_value,decay_steps=decay_steps,decay_rate=0.96)
    self.lr = tf.train.exponential_decay(
      self.init_value, self.global_step,
      decay_steps, decay_rate, staircase=True)

  @property
  def value(self):
    return self.lr

DefaultLearningRate = ExponentialDecay_LR
