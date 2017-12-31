'''Explore Weight Change

First need to run train_by_default.py to generate models that we used to explore.

Exploration:
1. The averaged weight changing in each layer between each epoch
2. The averaged weight changing in each layer before training and a well-trained network
3. Show the 5 most changed CNN kernels in each layer.
'''

import numpy as np
import matplotlib.pyplot as plt

