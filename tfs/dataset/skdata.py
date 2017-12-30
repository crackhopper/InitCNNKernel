import data_tool as dtool
import numpy as np
from base import Dataset
import sklearn.datasets as skdata

# TODO:
# the following is not wrapped
#
# 1. multilabel
# 2. manifold learning
# 3. decomposition
# 4. biclustering
#
# real dataset:
#
# - The Olivetti faces dataset
# - The 20 newsgroups text dataset
# - mldata.org repository
# - The Labeled Faces in the Wild face recognition dataset
# - Forest covertypes
# - RCV1 dataset
# - Boston House Prices dataset
# - Breast Cancer Wisconsin (Diagnostic) Database
# - Diabetes dataset
# - Optical Recognition of Handwritten Digits Data Set
# - Iris Plants Database
# - Linnerrud dataset

class SKData(Dataset):
  def __init__(self,**kwargs):
    self.setup(**kwargs)
    super(SKData,self).__init__(None)

  def setup(self):
    raise NotImplementedError()

  def load_train_test(self):
    n = self._x.shape[0]
    idx = np.arange(n)
    te_idx,tr_idx = dtool.split(idx,[self._test_percent])
    return self._x[tr_idx],self._y[tr_idx],self._x[te_idx],self._y[tr_idx]

class MakeBlobs(SKData):
  def __init__(self,**kwargs):
    super(MakeBlobs,self).__init__(**kwargs)

  def setup(self, test_percent = 0.3, n_samples=100, n_features=2, centers=3,
            cluster_std=1.0, center_box=(-10.0, 10.0), shuffle=True,
            random_state=None ):
    X, y = skdata.make_blobs(
      n_samples,
      n_features,
      centers,
      cluster_std,
      center_box,
      shuffle,
      random_state
    )
    self._test_percent = test_percent
    self._x = X
    self._y = y

class MakeClassification(SKData):
  def __init__(self,**kwargs):
    super(MakeClassification,self).__init__(**kwargs)

  def setup(self,test_percent = 0.3, n_samples=100, n_features=20,
            n_informative=2, n_redundant=2, n_repeated=0, n_classes=2,
            n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=1.0,
            hypercube=True, shift=0.0, scale=1.0, shuffle=True,
            random_state=None):
    X, y = skdata.make_classification(
      n_samples, n_features, n_informative, n_redundant,
      n_repeated, n_classes, n_clusters_per_class, weights, flip_y,
      class_sep, hypercube, shift, scale, shuffle, random_state
    )
    self._test_percent = test_percent
    self._x = X
    self._y = y

class MakeRegression(SKData):
  def __init__(self,**kwargs):
    super(MakeRegression,self).__init__(**kwargs)

  def setup(self,test_percent = 0.3, n_samples=100, n_features=100,
            n_informative=10, n_targets=1, bias=0.0, effective_rank=None,
            tail_strength=0.5, noise=0.0, shuffle=True, coef=False, random_state=None):
    X, y = skdata.make_regression(
      n_samples, n_features, n_informative, n_targets, bias, effective_rank,
      tail_strength, noise, shuffle, coef, random_state
    )
    self._test_percent = test_percent
    self._x = X
    self._y = y

class SKDataNoTest(Dataset):
  def __init__(self,**kwargs):
    self.setup(**kwargs)
    super(SKDataNoTest,self).__init__(None)

  def setup(self):
    raise NotImplementedError()

  def load_train_test(self):
    return self._x,self._y,self._x.copy(),self._y.copy()

class Iris(SKDataNoTest):
  def setup(self):
    iris = skdata.load_iris()
    self._x = iris.data
    self._y = iris.target

