from tfs.network.base import CustomNetwork
class LeNet(CustomNetwork):
  def setup(self):
    """http://ethereon.github.io/netscope/#/gist/87a0a390cff3332b476a
    Note : lr_mult parameter is different.
    """
    self.default_in_shape = [None,28,28,1]
    (self.net_def
     .conv2d([5,5],20,[1,1],activation=None,name='conv1',padding='VALID')
     .maxpool([2,2],[2,2],name='pool1',padding='VALID')
     .conv2d([5,5],50,[1,1],name='conv2',padding='VALID')
     .maxpool([2,2],[2,2],name='pool2',padding='VALID')
     .fc(500,name='ip1')
     .fc(10, activation=None,name='ip2')
     .softmax(name='prob')
    )
    self.loss_input_layer_name = 'ip2'
