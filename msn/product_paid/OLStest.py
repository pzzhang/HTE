from TVregression import TVregression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# load data
df = pd.read_csv("../data/msn_paid.csv", index_col=0)
values = df.values
product = list(values[20:31, 1])
gender = ['Female', 'Male']
RDelta = np.array(values[np.ix_(range(20, 31), [6, 12])], dtype=float)
VarRDelta = np.array(values[np.ix_(range(20, 31), [7, 13])], dtype=float)
nn = list(RDelta.shape)

# creat an object
chambo2 = TVregression(RDelta, 1.0/VarRDelta)
print "Initial Noise_Energy", chambo2.c/2
print "Initial Constant Term", chambo2.c0

# residual calculation
