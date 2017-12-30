from tfs.network.base import CustomNetwork
class CaffeNet(CustomNetwork):
  def setup(self):
    self.default_in_shape = [None,227,227,3]
    (self.net_def
     .conv2d([11,11],96, [4,4], padding='VALID', name='conv1') # (1, 55, 55, 96)
     .maxpool([3, 3], [2,2] , padding='VALID', name='pool1') # (1, 27, 27, 96)
     .lrn(2, 2e-05, 0.75, name='norm1')
     .conv2d([5,5],256, [1,1], group=2, name='conv2') #(1, 27, 27, 256)
     .maxpool([3, 3], [2, 2], padding='VALID', name='pool2') # (1, 13, 13, 256)
     .lrn(2, 2e-05, 0.75, name='norm2')
     .conv2d([3, 3], 384, [1, 1], name='conv3') # (1, 13, 13, 384)
     .conv2d([3, 3], 384, [1, 1], group=2, name='conv4')
     .conv2d([3, 3], 256, [1, 1], group=2, name='conv5') # (1, 13, 13, 256)
     .maxpool([3, 3], [2, 2], padding='VALID', name='pool5') # (1, 6, 6, 256)
     .fc(4096, name='fc6')
     .fc(4096, name='fc7')
     .fc(1000, activation=False, name='fc8')
     .softmax(name='prob'))
    self.loss_input_layer_name = 'fc8'

