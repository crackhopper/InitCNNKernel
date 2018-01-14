import numpy as np
def norm01(arr):
    arr = arr.copy()
    arr -= arr.min()
    arr /= arr.max() + 1e-10
    return arr


def norm01c(arr, center):
    '''Maps the input range to [0,1] such that the center value maps to .5'''
    arr = arr.copy()
    arr -= center
    arr /= max(2 * arr.max(), -2 * arr.min()) + 1e-10
    arr += .5
    assert arr.min() >= 0
    assert arr.max() <= 1
    return arr

def ensure_uint255(arr):
    '''If data is float, multiply by 255 and convert to uint8. Else leave as uint8.'''
    if arr.dtype == 'uint8':
        return arr
    elif arr.dtype in ('float32', 'float64'):
        #print 'extra check...'
        #assert arr.max() <= 1.1
        return np.array(arr * 255, dtype = 'uint8')
    else:
        raise Exception('ensure_uint255 expects uint8 or float input but got %s with range [%g,%g,].' % (arr.dtype, arr.min(), arr.max()))
from IPython.display import clear_output, Image, display, HTML
import numpy as np
# Helper functions for TF Graph visualization

def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = graph_def
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:800px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))
