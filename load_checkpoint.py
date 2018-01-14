from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('--eval_dir', type=str, default='../tfs_data/cifar10_eval',
                    help='Directory where to write event logs.')

parser.add_argument('--eval_data', type=str, default='test',
                    help='Either `test` or `train_eval`.')

parser.add_argument('--checkpoint_dir', type=str, default='../tfs_data/cifar10_train2',
                    help='Directory where to read model checkpoints.')

parser.add_argument('--eval_interval_secs', type=int, default=60*5,
                    help='How often to run the eval.')

parser.add_argument('--num_examples', type=int, default=10000,
                    help='Number of examples to run.')

parser.add_argument('--run_once', type=bool, default=False,
                    help='Whether to run eval only once.')

FLAGS = parser.parse_args()

IMAGE_SIZE = 32

from sklearn import preprocessing

from tfs.network import Network,CustomNetwork
from tfs.core.learning_rate import ExponentialDecay_LR
from tfs.core.optimizer import AdamOptimizer,GradientDecentOptimizer
from tfs.dataset import Cifar10
dataset = Cifar10()
dataset.transpose([0,2,3,1]) # for cpu testing, we use NHWC data format
dataset.standardize_per_sample()

class ConvNet(CustomNetwork):
  def setup(self):
    self.default_in_shape = [None,IMAGE_SIZE,IMAGE_SIZE,3]
    (self.net_def
     .conv2d([5, 5], 64, [1,1], name='conv1')
     .maxpool([3, 3], [2,2] , name='pool1')
     .lrn(4, 0.001/9.0, 0.75, bias=1.0, name='norm1')
     .conv2d([5,5], 64, [1,1], name='conv2')
     .maxpool([3, 3], [2, 2],  name='pool2')
     .lrn(4, 0.001/9.0, 0.75, bias=1.0, name='norm2')
     .fc(384, name='fc3')
     .fc(192, name='fc4')
     .fc(10, activation=False, name='fc5')
     .softmax(name='prob'))
    self.loss_input_layer_name = 'fc5'

net = ConvNet()
g = net.graph
net.build()

with g.as_default():
  ckptfile = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
  print(ckptfile)
  saver = tf.train.Saver()
  saver.restore(net.sess, ckptfile)

  logits = net.predict(dataset.test.data)
  top_k_op = tf.nn.in_top_k(logits, dataset.test.labels, 1)
  res = net.sess.run(top_k_op)

  print('accuracy score:',sum(res)/len(res))




