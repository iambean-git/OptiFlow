# v1.0.0

from IPython.display import set_matplotlib_formats, display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mglearn
from cycler import cycler
import matplotlib

matplotlib.rc("font", family="Malgun Gothic") # windows
# matplotlib.rc("font", family="AppleGothic") # mac
# matplotlib.rc('font', family='NanumBarunGothic') # Linux
matplotlib.rcParams["axes.unicode_minus"] = False # 한국어 쓰려면 무조건 바꿔야 해

# set_matplotlib_formats("pdf", "png")
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.dpi"] = 300
plt.rcParams["image.cmap"] = "viridis"
plt.rcParams["image.interpolation"] = "none"
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["legend.numpoints"] = 1
plt.rc(
    "axes",
    prop_cycle=(
        cycler("color", mglearn.plot_helpers.cm_cycle.colors)
        + cycler("linestyle", ["-", "-", "--", (0, (3, 3)), (0, (1.5, 1.5))])
    ),
)


np.set_printoptions(precision=3, suppress=True)

pd.set_option("display.max_columns", None)
pd.set_option("display.precision", 2)

__all__ = ["np", "mglearn", "display", "plt", "pd"]  
