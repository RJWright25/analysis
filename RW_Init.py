### RW_Init: basic presets

import matplotlib.pyplot as plt
import matplotlib.font_manager
import warnings
import numpy as np
import scipy
import h5py


warnings.filterwarnings("ignore")
### Plotting Parameters (matplotlib)
#### font presets

#legend
plt.rcParams['legend.fontsize'] = 10
#tuitle
plt.rcParams['figure.titlesize'] = 14
#ticks sizes
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
#axes labels
plt.rcParams['font.size'] = 14

#LaTeX
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['font.sans-serif'] = "Times"

#### image quality
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
