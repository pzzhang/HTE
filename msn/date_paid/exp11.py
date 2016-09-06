from TVregression import TVregression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# load data
df = pd.read_csv("../data/msn_paid.csv", index_col=0)
values = df.values
dates = values[1:16, 1]
RDelta = np.array(values[np.ix_(range(1, 16), [6, 12])], dtype=float)
VarRDelta = np.array(values[np.ix_(range(1, 16), [7, 13])], dtype=float)
nn = list(RDelta.shape)

# creat an object
chambo2 = TVregression(RDelta, 1.0/VarRDelta)
chambo2.set_graph(0, "linear")
print "Initial Noise_Energy", chambo2.c/2
print "Initial Constant Term", chambo2.c0

# least square solution
effect_ls = np.zeros((chambo2.nt1,chambo2.nt1))
labels = []
labelsx = []
cross_ind = 0
for d in range(chambo2.D):
    gind = range(chambo2.nind1[d], chambo2.nind1[d+1])
    effect_ls[gind, gind] = chambo2.u1ls[d]
    labeld = ["X%d:1" % (d+1)] + [str(i+1) for i in range(1, chambo2.n[d])]
    labels += labeld
    labelsx += [str(i+1) for i in range(chambo2.n[d])]
    for f in range(d+1, chambo2.D):
        ginf = range(chambo2.nind1[f], chambo2.nind1[f + 1])
        if not((d, f) == chambo2.pairs[cross_ind]):
            print "Something wrong in cross-term indexing!"
        effect_ls[np.ix_(gind, ginf)] = chambo2.u2ls[cross_ind]
        cross_ind += 1

# plot least square result
fig, ax = plt.subplots()
heatmap = ax.pcolor(effect_ls, cmap=plt.cm.Blues)
# put the major ticks at the middle of each cell
ax.set_xticks(np.arange(effect_ls.shape[0])+0.5, minor=False)
plt.xlim(0, chambo2.nt1)
ax.set_yticks(np.arange(effect_ls.shape[1])+0.5, minor=False)
plt.ylim(0, chambo2.nt1)
# want a more natural, table-like display
ax.invert_yaxis()
# ax.xaxis.tick_top()
# add colorbar
cbar = fig.colorbar(heatmap)
ax.set_xticklabels(labelsx, minor=False)
ax.set_yticklabels(labels, minor=False)
plt.title("Least square")
# plt.show()
plt.savefig('figure/LeastSquare.png')
plt.close(fig)

# get the TV settings
chambo2.get_K()
chambo2.set_w(Ns=10000)

# test pure lasso
u_lasso_path, noise_lasso_path, uls_lasso_path = chambo2.lasso_path(lam_min=1.5, lam_max=20.0, nstep=400)

# # test Bregman iteration
# lam = 0.01
# u_bregman, noise_bregman, uls_bregman = chambo2.bregman_iteration(lam=lam, nstep=2000)