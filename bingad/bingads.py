from TVreg1st import TVreg1st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# load data
df = pd.read_csv("data/volume.csv", index_col=0).fillna(0.0)
values = df.values
nn = [5, 2, 13, 7]
logdelta = np.array(values[:, 5].reshape(tuple(nn), order='C'), dtype=float)
mlogdelta = np.array(values[:, 7].reshape(tuple(nn), order='C'), dtype=float)

# creat an object
chambo2 = TVreg1st(logdelta, mlogdelta)
chambo2.set_graph(3, "circular")
print "Initial Noise_Energy", chambo2.c/2

# # least square solution
# labels = []
# for d in range(chambo2.D):
#     labeld = ["X%d:1" % (d + 1)] + [str(i + 1) for i in range(1, chambo2.n[d])]
#     labels += labeld
# # plot computed result
# fig, ax = plt.subplots()
# plt.plot(chambo2.uls, '-*')
# plt.xlim(0, len(chambo2.uls))
# ax.set_xticks(range(len(chambo2.uls)))
# ax.set_xticklabels(labels, minor=False)
# plt.title("LeastSquare-1st")
# # plt.show()
# plt.savefig('figure/firstorder/LeastSquare1st.png')
# plt.close(fig)

# get the TV settings
chambo2.get_K()
chambo2.set_w(Ns=10000)

# chambo2.P_PDALG1(Lambda=2.27, n_it=30000)

# # test TV-regression
# u_lasso_path, noise_lasso_path, uls_lasso_path = chambo2.lasso_path(lam_min=0.135, lam_max=3.0, nstep=300)
# u_lasso_path, noise_lasso_path, uls_lasso_path = chambo2.lasso_path(lam_min=1.78, lam_max=1.78, nstep=1)

# test Bregman iteration
lam = 0.02
u_bregman, noise_bregman, uls_bregman = chambo2.bregman_iteration(lam=lam, nstep=200)
