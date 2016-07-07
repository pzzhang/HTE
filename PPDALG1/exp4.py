from Chambolle0 import Chambolle0
from Chambolle import Chambolle
import numpy as np
import matplotlib.pyplot as plt


# generate data
nn = [10, 3, 5, 4]
D = len(nn)
# clean signal
u0_exact = 0.05
u1_exact = [np.zeros(nn[d]) for d in range(D)]
u1_exact[0][0:2] = np.array([0.03,-0.03])
u1_exact[1][1] = -0.02
# add noise
Ns = 10000
mm = [0.1*np.ones(10), np.array([0.2, 0.5, 0.3]), 0.2*np.ones(5), np.array([0.1, 0.4, 0.3, 0.2])]
m = mm[0]
u_exact = u0_exact + u1_exact[0]
m = m[:, np.newaxis]*mm[1]
u_exact = u_exact[:, np.newaxis]+u1_exact[1]
m = m[:, :, np.newaxis]*mm[2]
u_exact = u_exact[:, :, np.newaxis]+u1_exact[2]
m = m[:, :, :, np.newaxis]*mm[3]
u_exact = u_exact[:, :, :, np.newaxis]+u1_exact[3]
m *= Ns
v_exact = np.random.normal(loc=0, scale=0.1, size=m.shape)/np.sqrt(m)

g = u_exact + v_exact

# creat an object
# chambo0 = Chambolle0(g, m)
chambo = Chambolle(g, m)

# create the topology
# A = [None]*D
# for d in range(D):
#     A[d] = np.zeros((nn[d], nn[d]), dtype=bool)
#     A[d][range(1,nn[d]),range(0,nn[d]-1)] = True
#     A[d][range(0,nn[d]-1),range(1,nn[d])] = True
# d = 1
# A[d] = np.ones((nn[d], nn[d]), dtype=bool)
# A[d][range(0,nn[d]),range(0,nn[d])] = False
# chambo.set_graph(A)

# compute
# Lambda = 10. is good
Lambda = 200.0
n_it = 100000
# energy0, u00, u10 = chambo0.PDALG2(Lambda=Lambda, n_it=n_it)
energy1, u01, u11 = chambo.P_PDALG1(Lambda=Lambda, n_it=n_it)

# plot the result
fig = plt.figure()
for d in range(D):
    ax = plt.subplot(D, 1, d + 1)
    # plt.plot(range(1,nn[d]+1), u10[d], 'k^-', linewidth=2.0, label='recover_0')
    plt.plot(range(1, nn[d] + 1), u11[d], 'kv-', linewidth=2.0, label='TV-regression')
    plt.plot(range(1, nn[d] + 1), chambo.lssol[chambo.nind1[d]:chambo.nind1[d + 1]], 'bs-', label='simple-regression')
    plt.plot(range(1, nn[d] + 1), u1_exact[d], 'ro-', label='true')
    plt.ylim(-0.03, 0.03)
    ax.legend(loc='center left', bbox_to_anchor=(0.7, 0.8))
    plt.title("Covariate X%d, Lambda = %1.2e" % (d + 1, Lambda))
plt.show()
# plt.savefig('figure/exp4/Lambda = %1.2e_n_it = %1.2e.png' % (Lambda, n_it))
# plt.close(fig)
