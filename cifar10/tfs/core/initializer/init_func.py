import numpy as np
import tensorflow as tf
from tensorflow.python.ops import random_ops
import math
from tfs.core.elem import Param

def xavier(factor=1.0,mode='FAN_AVG',uniform=True,seed=None):
  p = Param(factor=factor,mode=mode,uniform=uniform,seed=seed)
  # pylint: disable=unused-argument
  def _initializer(shape, dtype):
    """Initializer function."""
    if not dtype.is_floating:
      raise TypeError('Cannot create initializer for non-floating point type.')
    # Estimating fan_in and fan_out is not possible to do perfectly, but we try.
    # This is the right thing for matrix multiply and convolutions.
    if shape:
      fan_in = float(shape[-2]) if len(shape) > 1 else float(shape[-1])
      fan_out = float(shape[-1])
    else:
      fan_in = 1.0
      fan_out = 1.0
    for dim in shape[:-2]:
      fan_in *= float(dim)
      fan_out *= float(dim)
    if mode == 'FAN_IN':
      # Count only number of input connections.
      n = fan_in
    elif mode == 'FAN_OUT':
      # Count only number of output connections.
      n = fan_out
    elif mode == 'FAN_AVG':
      # Average number of inputs and output connections.
      n = (fan_in + fan_out) / 2.0
    if uniform:
      # To get stddev = math.sqrt(factor / n) need to adjust for uniform.
      limit = math.sqrt(3.0 * factor / n)
      out= random_ops.random_uniform(shape, -limit, limit,
                                       dtype, seed=seed)
    else:
      # To get stddev = math.sqrt(factor / n) need to adjust for truncated.
      trunc_stddev = math.sqrt(1.3 * factor / n)
      out= random_ops.truncated_normal(shape, 0.0, trunc_stddev, dtype,
                                         seed=seed)
    out.for_print = 'xavier(%s)' % p
    return out
  # pylint: enable=unused-argument
  return _initializer



def constant(val=0.1):
  p = Param(val=val)
  def _init(shape,dtype):
    out = val*tf.ones(shape,dtype=dtype)
    out.for_print = 'constant(%s)' % p
    return out
  return _init


