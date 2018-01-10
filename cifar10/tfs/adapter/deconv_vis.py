from tfs.core.util import *
import tensorflow as tf
BKG_CLR = 0.95
def scale_0_1(array):
  _min = np.min(array)
  _max = np.max(array)
  return (array-_min)/(_max-_min)

def concate_tensor_to_image(tensor,n_sep=1):
  assert tensor.ndim == 3
  h = tensor.shape[0]
  w = tensor.shape[1]
  N = tensor.shape[2]

  n_w = int(np.ceil(np.sqrt(N)))
  n_h = (N-1)//n_w+1

  img_w = n_w*(w+n_sep)+n_sep
  img_h = n_h*(h+n_sep)+n_sep
  img = np.ones([img_h,img_w])*BKG_CLR

  for i in range(N):
    top_left_y = (i//n_w)*(h+n_sep)
    top_left_x = (i%n_w)*(w+n_sep)
    img[top_left_y+n_sep:top_left_y+h+n_sep,
        top_left_x+n_sep:top_left_x+w+n_sep]=tensor[...,i]
  return img

def prepare_tensor(tensor,dim_w,dim_h):
  tensor = scale_0_1(tensor)
  if tensor.ndim == 4:
    return redim_4_3(tensor,dim_w,dim_h)
  elif tensor.ndim == 2:
    return redim_2_3(tensor)
  else:
    raise RuntimeError('unsupported tensor dim')

def redim_4_3(tensor,dim_w,dim_h):
  assert tensor.ndim>=dim_w
  assert tensor.ndim>=dim_h
  dims = range(tensor.ndim)
  dims.remove(dim_w)
  dims.remove(dim_h)
  dims = [dim_w,dim_h]+dims
  tensor = tensor.transpose(dims)
  tensor = tensor.reshape([tensor.shape[0],tensor.shape[1],-1])
  return tensor

def redim_2_3(tensor):
  N = tensor.shape[0]
  n_channels = tensor.shape[1]
  size = int(np.ceil(np.sqrt(N)))
  t = np.ones([size*size,n_channels])*BKG_CLR
  t[0:N,:] = tensor
  return t.reshape([size,size,-1])

class DeconvVisNet(object):
  def __init__(self,netobj,nsamples=1):
    with netobj.graph.as_default():
      self.net = netobj
      netobj.nsamples = nsamples
      inv_in_shape = netobj._out.get_shape().as_list()
      self._inv_in = netobj._out
      tmp = self._inv_in
      for l in netobj.nodes[::-1]:
        tmp = l.inverse(tmp)
      self._inv_out = tmp
      layers = {}
      for l in netobj.nodes:
        layers[l.name]=l._out
      self.layers= layers

    self._cache = {}

  def _layer_output(self,layer_name,image):
    if layer_name in self._cache:
      return self._cache[layer_name]
    layer_output = self.net.sess.run(
      self.layers[layer_name],
      feed_dict={
        self.net._in:image
      })
    self._cache[layer_name] = layer_output
    return layer_output

  def img_deconv_channel(self,layer_name,channel_id,image):
    layer_output = self._layer_output(layer_name,image).copy()
    to_vis=np.zeros_like(layer_output)
    to_vis[0,...,channel_id]=layer_output[0,...,channel_id]

    generated = self.net.sess.run(
      self._inv_out,
      feed_dict={
        self.layers[layer_name]:to_vis,
        self.net._in:image
      })
    gen_img=ensure_uint255(norm01c(generated[0,:],0))
    img =gen_img[:,:,::-1]
    if img.shape[-1]==1:
      img = img[...,0]
    return img

  def img_layer(self,layer_name,image):
    layer_output = self._layer_output(layer_name,image).copy()
    layer_output = prepare_tensor(layer_output,1,2)
    img = concate_tensor_to_image(layer_output)
    return img


  def img_weight(self,layer_name,image):
    l = self.net.node_by_name(layer_name)
    assert 'weights' in l.variables
    v = l.variables['weights']
    res = self.net.run(v)
    res = prepare_tensor(res,0,1)
    img = concate_tensor_to_image(res)
    return img
