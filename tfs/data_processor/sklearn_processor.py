from sklearn import preprocessing
from base import *
from tfs.dataset.base import DataSubset

class _SKLearnType(object):
  undecide = 0
  data = 1
  labels = 2
  both = 3

class SKLearnTransformer(BaseProcessor):
  _type=_SKLearnType.undecide
  def _apply(self,op,dataset):
    if self._type ==_SKLearnType.undecide:
      raise RuntimeError("%s does not define _type variable"%type(self).__name__)
    elif self._type ==_SKLearnType.data:
      res = op(dataset.data)
      return DataSubset(res,dataset.labels)
    elif self._type ==_SKLearnType.labels:
      res=op(dataset.labels)
      return DataSubset(dataset.data,res)
    elif self._type ==_SKLearnType.both:
      res=op(dataset.data,dataset.labels)
      return DataSubset(res,dataset.labels)
    else:
      raise RuntimeError("%s define an unsupported _type variable"%type(self).__name__)

  def fit_transform(self,dataset):
    return self._apply(self.p.fit_transform,dataset)

  def transform(self,dataset):
    return self._apply(self.p.transform,dataset)

  def inverse_transform(self,dataset):
    return self._apply(self.p.inverse_transform,dataset)

class LabelBinarizer(SKLearnTransformer):
  """ a wrapper for sklearn.preprocessing.LabelBinarizer
  see http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html
  """
  _type=_SKLearnType.labels
  def __init__(self,neg_label=0, pos_label=1, sparse_output=False):
    self.p = preprocessing.LabelBinarizer(neg_label, pos_label, sparse_output)

class StandardScaler(SKLearnTransformer):
  """ a wrapper for sklearn.preprocessing.StandardScaler
  see http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
  """
  _type=_SKLearnType.data
  def __init__(self,copy=True, with_mean=True, with_std=True):
    self.p = preprocessing.StandardScaler(copy, with_mean, with_std)

class MinMaxScaler(SKLearnTransformer):
  """ a wrapper for sklearn.preprocessing.MinMaxScaler
  see http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
  """
  _type=_SKLearnType.data
  def __init__(self,feature_range=(0, 1), copy=True):
    self.p = preprocessing.MinMaxScaler(feature_range, copy)

class Normalizer(SKLearnTransformer):
  """ a wrapper for sklearn.preprocessing.Normalizer
  see http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html
  """
  _type=_SKLearnType.data
  def __init__(self,norm='l2', copy=True):
    self.p = preprocessing.Normalizer(norm,copy)
