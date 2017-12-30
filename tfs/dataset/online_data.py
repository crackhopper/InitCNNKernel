from six.moves import urllib
from base import Dataset,DataSubset
import data_tool as dtool
import numpy as np
import os
import tfs.g
import tarfile
import gzip
import shutil
import sys

def _tar_gz_extractor(f,d):
  tarfile.open(f, 'r:gz').extractall(d)

def _gz_extractor(f,d):
  fname = os.path.split(f)[-1]
  foutpath = os.path.join(d,os.path.splitext(fname)[0])
  with gzip.open(f,'rb') as f_in:
    with open(foutpath,'wb') as f_out:
      f_out.writelines(f_in)

def _just_copy(f,d):
  shutil.copy(f,os.path.join(d,f))

_extractors={
  '.tar.gz': _tar_gz_extractor,
  '.gz':_gz_extractor,
  '':_just_copy,
}

def _get_extractor(fname):
  basename = os.path.split(fname)[1]
  exts=basename.split('.')
  if len(exts)==1:
    return _extractors['']
  exts=exts[1:]
  for i in range(len(exts)):
    ext = '.'+'.'.join(exts[i:])
    if ext in _extractors:
      return _extractors[ext]
  return _extractors['']



class Online(Dataset):
  urls = {
  }
  filelists=[]
  def prepare(self):
    if self.data_dir is None:
      self.data_dir = tfs.g.config.dataset.base_dir+self.default_dir
    if self.check_files():
      return
    self.maybe_download()
    self.extract()

  def check_files(self):
    return dtool.check_exist([os.path.join(self.data_dir,f) for f in self.filelists])

  def maybe_download(self):
    download_dir = tfs.g.config.dataset.download_dir
    if not os.path.exists(download_dir):
      os.makedirs(download_dir)
    for f,link in self.urls.items():
      filepath = os.path.join(download_dir, f)
      if link is None:
        raise ValueError("Cannot get the file %s from %s"%(f,link))
      if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
          sys.stdout.write('\r>> Downloading %s %.1f%%' % (
            f,
            float(count * block_size) / float(total_size) * 100.0))
          sys.stdout.flush()

        print ''
        filepath, _ = urllib.request.urlretrieve(link, filepath,
                                                 reporthook=_progress)
        print ''
        statinfo = os.stat(filepath)
        print 'Successfully downloaded', f, statinfo.st_size, 'bytes.'


  def extract(self):
    if not os.path.exists(self.data_dir):
      os.makedirs(self.data_dir)
    for f in self.urls:
      filepath = os.path.join(tfs.g.config.dataset.download_dir,f)
      extractor = _get_extractor(filepath)
      extractor(filepath,self.data_dir)
      print 'extracting ',filepath, ' into ',self.data_dir

