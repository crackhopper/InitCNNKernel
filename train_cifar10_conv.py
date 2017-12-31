from tfs.core.monitor import SaveEpochMonitor
from tfs.models import ConvNet
from tfs.dataset import Cifar10
from tfs.core.learning_rate import ExponentialDecay_LR
from tfs.core.optimizer import AdamOptimizer
modelfile = './models/cifar10-convnet'
net = ConvNet() # this is a network structure similar with AlexNet
dataset = Cifar10()
dataset.transpose([0,2,3,1]) # for cpu testing, we use NHWC data format

epoch = 100
bsize = 500
net.add_monitor('SaveNet',SaveEpochMonitor(net,modelfile))

net.lr = ExponentialDecay_LR(net,0.001,0,500,0.9)
net.optimizer = AdamOptimizer(net)
net.build()

net.fit(dataset,batch_size=bsize,n_epoch=epoch)
net.save(modelfile+'_%d'%epoch)


