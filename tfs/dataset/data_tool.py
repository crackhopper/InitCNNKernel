from __future__ import division
import numpy as np
import tensorflow as tf

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def check_exist_raise(filenames):
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

def check_exist(filenames):
  for f in filenames:
    if not tf.gfile.Exists(f):
      return False
  return True

def split(arr,percents,shuffle=True,dim=0):
  """split data into several set:
  percent : list of float numbers, and sum(percent)<1
  return : arr_1, ..., arr_n.  number of each contain corresponding percent of the data
  """
  n = arr.shape[dim]

  all_p = np.sum(np.array(percents))
  assert all_p <= 1
  if not isclose(all_p,1):
    percents.append(1-all_p)

  curs = []
  cur=0
  for p in percents:
    curs.append(cur)
    nn = int(n*p)
    cur += nn
  curs.append(n)

  idx = range(n)
  if shuffle:
    np.random.shuffle(idx)

  res = []
  slices = [slice(None,None,None)]*len(arr.shape)
  for i in range(len(percents)):
    _idx = idx[curs[i]:curs[i+1]]
    slices[dim]=_idx
    res.append(arr[slices])

  return res

def split_n(arr,n,shuffle=True,dim=0):
  ps = [1.0/n]*n
  return split(arr,ps,shuffle,dim=0)
