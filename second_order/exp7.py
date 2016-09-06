from Chambolle0 import Chambolle0
from Chambolle import Chambolle
from Chambolle2 import Chambolle2
# from Chambolle2_new import Chambolle2
import numpy as np
import matplotlib.pyplot as plt


# generate data
nn = [10, 3, 5, 4]
D = len(nn)
# clean signal
u0_exact = 0.05
u1_exact = [np.zeros(nn[d]) for d in range(D)]
u1_exact[0][0:2] = np.array([0.05, -0.05])
u1_exact[1][1] = -0.04
u2_exact = []
for d in range(D - 1):
    for f in range(d + 1, D):
        u2_exact.append(np.zeros((nn[d],nn[f])))
u2_exact[0][0,1] += 0.1
# add noise
Ns = 1e4
mm = [0.1*np.ones(10), np.array([0.2, 0.5, 0.3]), 0.2*np.ones(5), np.array([0.1, 0.4, 0.3, 0.2])]
# add zero-th and first order effects
m = mm[0]
u_exact = u0_exact + u1_exact[0]
m = m[:, np.newaxis]*mm[1]
u_exact = u_exact[:, np.newaxis]+u1_exact[1]
m = m[:, :, np.newaxis]*mm[2]
u_exact = u_exact[:, :, np.newaxis]+u1_exact[2]
m = m[:, :, :, np.newaxis]*mm[3]
u_exact = u_exact[:, :, :, np.newaxis]+u1_exact[3]
# add second order effect
u_exact += u2_exact[0][:, :, np.newaxis, np.newaxis]
m *= Ns
v_exact = np.random.normal(loc=0, scale=0.1, size=m.shape)/np.sqrt(m)

g = u_exact + v_exact

# creat an object
# chambo0 = Chambolle0(g, m)
# chambo = Chambolle(g, m)
chambo2 = Chambolle2(g, m)

# least square solution
effect_exact = np.zeros((chambo2.nt1,chambo2.nt1))
effect_ls = np.zeros((chambo2.nt1,chambo2.nt1))
labels = []
cross_ind = 0
for d in range(chambo2.D):
    gind = range(chambo2.nind1[d], chambo2.nind1[d+1])
    effect_exact[gind, gind] = u1_exact[d]
    effect_ls[gind, gind] = chambo2.u1ls[d]
    labels += ["X%d:%d" % (d+1, i+1) for i in range(chambo2.n[d])]
    for f in range(d+1, chambo2.D):
        ginf = range(chambo2.nind1[f], chambo2.nind1[f + 1])
        if not((d, f) == chambo2.pairs[cross_ind]):
            print "Something wrong in cross-term indexing!"
        effect_exact[np.ix_(gind, ginf)] = u2_exact[cross_ind]
        effect_ls[np.ix_(gind, ginf)] = chambo2.u2ls[cross_ind]
        cross_ind += 1

# get the objects
chambo2.get_K()
chambo2.set_w(Ns=10000)

# compute
# Lambda = 10. is good
Lambda = 100.0
n_it = 100000
# energy0, u00, u10 = chambo0.PDALG2(Lambda=Lambda, n_it=n_it)
# energy1, u01, u11 = chambo.P_PDALG1(Lambda=Lambda, n_it=n_it)
energy2, u12, u22 = chambo2.P_PDALG1(Lambda=Lambda, n_it=n_it)

# plot the result
fig = plt.figure()
plt.plot(range(n_it), energy2-energy2[-1], 'k-', label='P_PDALG')
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='best')
plt.title("Iteration v.s. Energy")
# plt.show()
plt.savefig('figure/exp7/EnergyDecay_Lambda = %1.2e_n_it = %1.2e.png' % (Lambda, n_it))
plt.close(fig)

fig = plt.figure()
for d in range(D):
    ax = plt.subplot(D, 1, d + 1)
    # plt.plot(range(1,nn[d]+1), u10[d], 'k^-', linewidth=2.0, label='recover_0')
    plt.plot(range(1, nn[d] + 1), u12[d], 'kv-', linewidth=2.0, label='TV-regression')
    plt.plot(range(1, nn[d] + 1), chambo2.u1ls[d], 'bs-', label='simple-regression')
    plt.plot(range(1, nn[d] + 1), u1_exact[d], 'ro-', label='true')
    ax.legend(loc='center left', bbox_to_anchor=(0.7, 0.8))
    plt.title("Covariate X%d, Lambda = %1.2e" % (d + 1, Lambda))
# plt.show()
plt.savefig('figure/exp7/FirstOrder_Lambda = %1.2e_n_it = %1.2e.png' % (Lambda, n_it))
plt.close(fig)

# first and second order effect
# patch all data into a big matrix
effect = np.zeros((chambo2.nt1,chambo2.nt1))
cross_ind = 0
for d in range(chambo2.D):
    gind = range(chambo2.nind1[d], chambo2.nind1[d+1])
    effect[gind, gind] = u12[d]
    for f in range(d+1, chambo2.D):
        ginf = range(chambo2.nind1[f], chambo2.nind1[f + 1])
        if not((d, f) == chambo2.pairs[cross_ind]):
            print "Something wrong in cross-term indexing!"
        effect[np.ix_(gind, ginf)] = u22[cross_ind]
        cross_ind += 1

# plot ground truth
fig, ax = plt.subplots()
heatmap = ax.pcolor(effect_exact, cmap=plt.cm.Blues)
# put the major ticks at the middle of each cell
ax.set_xticks(np.arange(effect_exact.shape[0]) + 0.5, minor=False)
plt.xlim(0, chambo2.nt1)
ax.set_yticks(np.arange(effect_exact.shape[1]) + 0.5, minor=False)
plt.ylim(0, chambo2.nt1)
# want a more natural, table-like display
ax.invert_yaxis()
ax.xaxis.tick_top()
# add colorbar
cbar = fig.colorbar(heatmap)
ax.set_xticklabels(labels, minor=False)
ax.set_yticklabels(labels, minor=False)
plt.title("Ground Truth")
# plt.show()
plt.savefig('figure/exp7/GroundTruth.png')
plt.close(fig)

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
ax.xaxis.tick_top()
# add colorbar
cbar = fig.colorbar(heatmap)
ax.set_xticklabels(labels, minor=False)
ax.set_yticklabels(labels, minor=False)
plt.title("Least square")
# plt.show()
plt.savefig('figure/exp7/LeastSquare.png')
plt.close(fig)

# plot computed result
fig, ax = plt.subplots()
heatmap = ax.pcolor(effect, cmap=plt.cm.Blues)
# put the major ticks at the middle of each cell
ax.set_xticks(np.arange(effect.shape[0])+0.5, minor=False)
plt.xlim(0, chambo2.nt1)
ax.set_yticks(np.arange(effect.shape[1])+0.5, minor=False)
plt.ylim(0, chambo2.nt1)
# want a more natural, table-like display
ax.invert_yaxis()
ax.xaxis.tick_top()
# add colorbar
cbar = fig.colorbar(heatmap)
ax.set_xticklabels(labels, minor=False)
ax.set_yticklabels(labels, minor=False)
plt.title("TV regularized regression")
# plt.show()
plt.savefig('figure/exp7/SecondOrder_Lambda = %1.2e_n_it = %1.2e.png' % (Lambda, n_it))
plt.close(fig)