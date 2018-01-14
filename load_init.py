from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def get_all_checkpointfiles(checkpoint_dir='../tfs_data/cifar10_train2'):
  state=tf.train.get_checkpoint_state(checkpoint_dir)
  return state.all_model_checkpoint_paths

def get_init_checkpointfiles(checkpoint_dir='../tfs_data/cifar10_train2'):
  return get_all_checkpointfiles(checkpoint_dir)[0]

# run after build the net
def load_into_net(net,ckptfile):
  g = tf.Graph()
  tbl = {}
  with g.as_default():
    with tf.Session() as sess:
      saver = tf.train.import_meta_graph(ckptfile+'.meta')
      saver.restore(sess, ckptfile)
      def fill_table(key):
        tbl[key] = sess.run(g.get_tensor_by_name(key))
      fill_table('conv1/weights:0')
      fill_table('conv1/biases:0')
      fill_table('conv2/weights:0')
      fill_table('conv2/biases:0')
      fill_table('fc3/weights:0')
      fill_table('fc3/biases:0')
      fill_table('fc4/weights:0')
      fill_table('fc4/biases:0')
      fill_table('fc5/weights:0')
      fill_table('fc5/biases:0')
  init_op=net.initializer.op_by_value_table(tbl)
  net.run(init_op)

