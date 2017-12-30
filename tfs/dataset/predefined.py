from online_data import Online
import data_tool as dtool
import numpy as np
import os

class Mnist(Online):
  """https://www.cs.toronto.edu/~kriz/cifar.html
  """
  baseurl='http://yann.lecun.com/exdb/mnist/'
  urls = {
    'train-images-idx3-ubyte.gz':baseurl+'train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz':baseurl+'train-labels-idx1-ubyte.gz',
    't10k-images-idx3-ubyte.gz':baseurl+'t10k-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz':baseurl+'t10k-labels-idx1-ubyte.gz'
  }
  filelists=[
    'train-images-idx3-ubyte',
    'train-labels-idx1-ubyte',
    't10k-images-idx3-ubyte',
    't10k-labels-idx1-ubyte'
  ]
  default_dir='mnist'
  def load_train_test(self):
    trX,trY = self.load(*self.filelists[0:2])
    teX,teY = self.load(*self.filelists[2:4])
    return trX,trY,teX,teY

  def load(self,f_data,f_label):
    def _read32(bytestream):
      dt = np.dtype(np.uint32).newbyteorder('>')
      return np.frombuffer(bytestream.read(4), dtype=dt)[0]

    f_data = self.data_full_path(f_data)
    f_label = self.data_full_path(f_label)

    with open(f_data,'rb') as bytestream:
      magic = _read32(bytestream)
      if magic != 2051:
        raise ValueError(
          'Invalid magic number %d in MNIST image file: %s' %
          (magic, f_data))
      num_images = _read32(bytestream)
      rows = _read32(bytestream)
      cols = _read32(bytestream)
      buf = bytestream.read(rows * cols * num_images)
      data = np.frombuffer(buf, dtype=np.uint8)
      data = data.reshape(num_images, rows, cols, 1)

    with open(f_label,'rb') as bytestream:
      magic = _read32(bytestream)
      if magic != 2049:
        raise ValueError(
          'Invalid magic number %d in MNIST label file: %s' %
          (magic, l_data))
      num_items = _read32(bytestream)
      buf = bytestream.read(num_items)
      labels = np.frombuffer(buf, dtype=np.uint8)
    return data,labels

class Cifar10(Online):
  """https://www.cs.toronto.edu/~kriz/cifar.html
  """
  urls = {
    "cifar-10-python.tar.gz":"https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
  }
  filelists=['cifar-10-batches-py/data_batch_%d' % i for i in xrange(1, 6)]+['cifar-10-batches-py/test_batch']
  default_dir='cifar10'
  def load_train_test(self):
    trX,trY = self.load(self.filelists[0:5])
    teX,teY = self.load([self.filelists[-1]])
    return trX,trY,teX,teY

  width=32
  height=32
  channels=3
  def load(self,filenames):
    def unpickle(file):
      import cPickle
      fo = open(file, 'rb')
      dict = cPickle.load(fo)
      fo.close()
      return dict
    data = []
    labels = []
    for f in filenames:
      batch=unpickle(os.path.join(self.data_dir,f))
      data.append(batch['data'])
      labels.append(batch['labels'])
    data = np.concatenate(data)
    labels = np.concatenate(labels)
    data = data.reshape([-1,self.channels,self.width,self.height])
    return data,labels

