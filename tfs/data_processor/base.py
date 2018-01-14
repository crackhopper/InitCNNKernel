class BaseProcessor(object):
  def fit_transform(self,dataset):
    raise NotImplementedError("implement by subclass")
  def transform(self,dataset):
    raise NotImplementedError("implement by subclass")
  def inverse_transform(self):
    raise NotImplementedError("implement by subclass")

class Pipeline(BaseProcessor):
  def __init__(self,processors):
    self.processors = processors
  def fit_transform(self,dataset):
    tmp = dataset
    for p in self.processors:
      tmp = p.fit_transform(tmp)
    return tmp
  def transform(self,dataset):
    tmp = dataset
    for p in self.processors:
      tmp = p.transform(tmp)
    return tmp
  def inverse_transform(self):
    tmp = dataset
    for p in self.processors[::-1]:
      tmp = p.inverse_transform(tmp)
    return tmp

