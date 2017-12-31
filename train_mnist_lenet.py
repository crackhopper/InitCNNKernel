## Mnist+LeNet Training
from tfs.core.monitor import SaveEpochMonitor
from tfs.models import LeNet
from tfs.dataset import Mnist

epoch = 10
bsize = 500
modelfile = './models/mnist-lenet'
net = LeNet()
dataset = Mnist()

net.add_monitor('SaveNet',SaveEpochMonitor(net,modelfile))
net.build()

net.fit(dataset,batch_size=bsize,n_epoch=epoch)
net.save(modelfile+'_%d'%epoch)

