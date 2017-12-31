import numpy as np
class Monitor(object):
  def __init__(self,netobj,interval=10):
    self.net = netobj
    self.interval = interval
    self.init()

  def init(self):
    self.results = []
    self.steps = []
    self.epochs = []

  def check(self,train,test,step,epoch):
    return True

  def status(self,train,test,step,epoch):
    if step % self.interval==0:
      if(self.check(train,test,step,epoch)):
        self.work(train,test,step,epoch)
        self.record(train,test,step,epoch)

  def work(self,train,test,step,epoch):
    raise NotImplementedError

  def record(self,train,test,step,epoch):
    self.steps.append(step)
    self.epochs.append(epoch)

  def __repr__(self):
    return '%s(net,interval=%d)'%(type(self).__name__,self.interval)

class DefaultMonitor(Monitor):
  def work(self,train,test,step,epoch):
    acc = self.net.score(test)
    X,y = train.current_batch
    loss = self.net.measure_loss(X,y)
    lr = self.net.lr.value
    print 'lr:%s, step %d. loss %f, score:%f'%(lr,step,loss,acc)


class LayerInputVarMonitor(Monitor):
  def work(self,train,test,step,epoch):
    X,y = train.current_batch
    in_tensors = []
    for n in self.net.nodes:
      in_tensors.append(n.input)
    vars = self.net.run(in_tensors,feed_dict={self.net.input:X})
    vars = [np.var(v) for v in vars]
    self.results.append(vars)


class SaveEpochMonitor(Monitor):
  def __init__(self,netobj,filename,interval=10):
    super(SaveEpochMonitor,self).__init__(netobj,interval)
    self.lastepoch = -1
    self.filename  = filename

  def check(self,train,test,step,epoch):
    if self.lastepoch == epoch:
      return False
    else:
      self.lastepoch = epoch
      return True

  def work(self,train,test,step,epoch):
    fname = self.filename+'_%d'%epoch
    print 'saving model into',fname
    self.net.save(fname)
