from .utils import *
from .core import *
import matplotlib.pyplot as plt

cm = 1/2.54 # inch/cm conversion factor
page = (21.0 - 1.65 - 1.25)*cm # A4 page

# matplotlib stylesheet
plt.style.use([pkg_resources.resource_filename('HydroToolkit', 'style/thesis.mplstyle')])

package_version = "0.1"