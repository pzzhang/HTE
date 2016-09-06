import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# load data
df = pd.read_csv("data/rpmdate.csv", index_col=0).replace([np.inf, -np.inf], np.nan)
df = df.fillna(0.0)
df.plot()
plt.show()

