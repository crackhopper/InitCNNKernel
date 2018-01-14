from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

IMAGE_SIZE = 32

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.
batch_size=128
log_frequency = 10
data_dir= '../tfs_data/cifar10/cifar-10-batches-bin'
train_dir= '../tfs_data/cifar10_train0'

def read_cifar10(filename_queue):
  class CIFAR10Record(object):
    pass
  result = CIFAR10Record()

  label_bytes = 1  # 2 for CIFAR-100
  result.height = 32
  result.width = 32
  result.depth = 3
  image_bytes = result.height * result.width * result.depth
  record_bytes = label_bytes + image_bytes
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)

  result.key, value = reader.read(filename_queue)
  record_bytes = tf.decode_raw(value, tf.uint8)


  result.label = tf.cast(
      tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

  depth_major = tf.reshape(
      tf.strided_slice(record_bytes, [label_bytes],
                       [label_bytes + image_bytes]),
      [result.depth, result.height, result.width])

  result.uint8image = tf.transpose(depth_major, [1, 2, 0])
  return result

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    shuffle):
  num_preprocess_threads = 16
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  return images, tf.reshape(label_batch, [batch_size])

def distorted_inputs():
  filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
               for i in xrange(1, 6)]
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_cifar10(filename_queue)

  reshaped_image = tf.cast(read_input.uint8image, tf.float32)
  height = IMAGE_SIZE
  width = IMAGE_SIZE

  distorted_image = tf.random_crop(reshaped_image, [height, width, 3])
  distorted_image = tf.image.random_flip_left_right(distorted_image)
  distorted_image = tf.image.random_brightness(distorted_image,
                                               max_delta=63)
  distorted_image = tf.image.random_contrast(distorted_image,
                                             lower=0.2, upper=1.8)

  float_image = tf.image.per_image_standardization(distorted_image)
  float_image.set_shape([height, width, 3])
  read_input.label.set_shape([1])

  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d CIFAR images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, 
                                         shuffle=True)

def inputs(eval_data):
  if not eval_data:
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                 for i in xrange(1, 6)]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  else:
    filenames = [os.path.join(data_dir, 'test_batch.bin')]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_cifar10(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # Image processing for evaluation.
  # Crop the central [height, width] of the image.
  resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                         height, width)
  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_standardization(resized_image)

  # Set the shapes of tensors.
  float_image.set_shape([height, width, 3])
  read_input.label.set_shape([1])

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, 
                                         shuffle=False)


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

def get_train_op():
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
  net.lr = ExponentialDecay_LR(net,INITIAL_LEARNING_RATE,0,decay_steps,LEARNING_RATE_DECAY_FACTOR)
  net.optimizer = GradientDecentOptimizer(net)
  net.build()
  return net._get_train_op()

from sklearn import preprocessing

def train():
  with net._graph.as_default():
    global_step = tf.train.get_or_create_global_step()
    with tf.device('/cpu:0'):
      images, labels = distorted_inputs()

    train_op = get_train_op()

    class _LoggerHook(tf.train.SessionRunHook):
      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(lss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = log_frequency * batch_size / duration
          sec_per_batch = float(duration / log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))

    with tf.Session() as sess:
      net._initialize()
      vars = net.optimizer.variables
      net.run(tf.variables_initializer(vars.values()))

      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess,coord=coord)

      i,l = sess.run([images,labels])
      lb = preprocessing.LabelBinarizer(0, 1, False)
      lb.fit(l)

      coord.request_stop()
      coord.join(threads)

    with net.sess as sess:
      # use official initializer
      from load_init import get_init_checkpointfiles,load_into_net
      ckpt = get_init_checkpointfiles()
      load_into_net(net,ckpt)

      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess,coord=coord)

      count = 0
      while True:
        count +=1
        i,l = sess.run([images,labels])
        l2 = lb.transform(l)
        lss = net.loss
        lossval,_ = sess.run([lss,train_op],feed_dict={net.input:i,net.true_output:l2})
        if(count%10==0):
          print('step:%d, loss:'%count,lossval)

        if(count%100==0):
          print('accuracy score:',net.score2(dataset.test,lb))

      coord.request_stop()
      coord.join(threads)

train()

'''
with tf.device('/cpu:0'):
  images, labels = distorted_inputs()
with tf.Session() as sess:
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess,coord=coord)

  i,l = sess.run([images,labels])
  print(i.shape,l.shape)

  coord.request_stop()
  coord.join(threads)

'''
