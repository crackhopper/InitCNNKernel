
from conv import *
from fc import *
from inference import Softmax
from normalization import LRN, BN
from pool import MaxPool,AvgPool
from dropout import Dropout

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
