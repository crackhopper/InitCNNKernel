from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tfs.core.layer.conv import *
from tfs.core.layer.fc import *
from tfs.core.layer.inference import Softmax
from tfs.core.layer.normalization import LRN, BN
from tfs.core.layer.pool import MaxPool,AvgPool
from tfs.core.layer.dropout import Dropout

func_table = {
  'conv2d':Conv2d,
  'fc':FullyConnect,
  'softmax':Softmax,
  'lrn':LRN,
  'bn':BN,
  'maxpool':MaxPool,
  'avgpool':AvgPool,
  'dropout':Dropout
}
