from tfs.core.monitor import SaveEpochMonitor,ScoreMonitor
from tfs.models import ConvNet
from tfs.dataset import Cifar10
from tfs.core.learning_rate import ExponentialDecay_LR
from tfs.core.optimizer import AdamOptimizer
modelfile = './models/cifar10-convnet'
net = ConvNet() # this is a network structure similar with AlexNet
dataset = Cifar10()
dataset.transpose([0,2,3,1]) # for cpu testing, we use NHWC data format
dataset.standardize_per_sample()

epoch = 200
bsize = 128

net.add_monitor('SaveNet',SaveEpochMonitor(net,modelfile,epoch_interval=10))
net.add_monitor('ScoreNet',ScoreMonitor(net))
net.lr = ExponentialDecay_LR(net,0.1,0,100000,0.1)
net.optimizer = AdamOptimizer(net)
net.build()

net.fit(dataset,batch_size=bsize,n_epoch=epoch)
net.save(modelfile+'_%d'%epoch)


