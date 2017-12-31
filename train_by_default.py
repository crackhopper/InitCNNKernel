'''Training models for further exploration

Experiment with : 1. Mnist+LeNet, 2. Cifar10+ConvNet, 3. ImageNet2012+(AlexNet, ResNet)
'''

from tfs.core.monitor import SaveEpochMonitor
epoch = 10
bsize = 500

## Mnist+LeNet Training
from tfs.models import LeNet
from tfs.dataset import Mnist
modelfile = './models/mnist-lenet'
net = LeNet()
dataset = Mnist()

net.add_monitor('SaveNet',SaveEpochMonitor(net,modelfile))
net.build()

net.fit(dataset,batch_size=bsize,n_epoch=epoch)
net.save(modelfile+'_%d'%epoch)
del net

## Cifar10+ConvNet Training
from tfs.models import ConvNet
from tfs.dataset import Cifar10
modelfile = './models/cifar10-convnet'
net = ConvNet() # this is a network structure similar with AlexNet
dataset = Cifar10()
dataset.transpose([0,2,3,1]) # for cpu testing, we use NHWC data format

net.add_monitor('SaveNet',SaveEpochMonitor(net,modelfile))
net.build()

net.fit(dataset,batch_size=bsize,n_epoch=epoch)
net.save(modelfile+'_%d'%epoch)
del net
