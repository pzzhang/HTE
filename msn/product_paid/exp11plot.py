from TVregression import TVregression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# load data
df = pd.read_csv("../data/msn_paid.csv", index_col=0)
values = df.values
product = values[20:31, 1]
RDelta = np.array(values[np.ix_(range(20, 31), [6, 12])], dtype=float)
VarRDelta = np.array(values[np.ix_(range(20, 31), [7, 13])], dtype=float)
# VarRDelta[:, 0] = 1.0*VarRDelta[:, 1]
VarRDelta[-2, 0] = 1.0*VarRDelta[0, 1]
# VarRDelta[:, :] = VarRDelta[0, 1]

# plot error bar
x = np.arange(len(product))
yf = RDelta[:, 0]
ym = RDelta[:, 1]
ef = 1.96*np.sqrt(VarRDelta[:, 0])
em = 1.96*np.sqrt(VarRDelta[:, 1])

fig, ax = plt.subplots()
plt.errorbar(x, yf, ef, marker='^', ecolor='b', label='Female')
plt.errorbar(x, ym, em, marker='v', ecolor='g', label='Male')
plt.plot(x, np.zeros(len(product)), 'r')
plt.legend(loc='best')
plt.xlim([-1, len(product)])
#plt.ylim([-1.5, 0.5])
ax.set_xticks(np.arange(len(product)), minor=False)
ax.set_xticklabels(product, minor=False, rotation=45)
plt.ylabel('Delta%')
plt.title("Purchase per User")
fig.autofmt_xdate()

plt.show()