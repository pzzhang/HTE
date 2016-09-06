import numpy as np
import pandas as pd
import cvxopt
# disable GLPK output
cvxopt.solvers.options['LPX_K_MSGLEV'] = 0         # old versions of cvxopt
cvxopt.solvers.options['msg_lev'] = 'GLP_MSG_OFF'  # works on cvxopt 1.1.7


class Chambolle2(object):
    def __init__(self, g, m):
        # g and m are D-dimensional arrays with the same shape
        # g is the point-wise averaged noisy measurement
        # m is the effective percentage of measurements at the corresponding point (inverse of variance)
        #   min_u \|\grad u\|_{1} + Lambda/2 \|u - g\|_{2,m}^2
        self.g = g
        self.n = np.array(g.shape)
        self.D = len(self.n)
        self.N = np.prod(self.n)
        self.m = m/(1.0*self.N)
        self.msum = np.sum(self.m)

        # the adjacency matrix A and the degree (on each level) at each node nlevel
        # default: fully connected
        self.A = [np.ones((self.n[d], self.n[d]+1), dtype=bool) for d in range(self.D)]
        for d in range(self.D):
            self.A[d][range(0, self.n[d]), range(0, self.n[d])] = False
        self.nlevel = [np.sum(self.A[d], axis=1) for d in range(self.D)]

        # total degree of freedom and parameter index for 1st order method
        self.nind1 = [0]
        for d in range(self.D):
            self.nind1.append(self.nind1[-1] + self.n[d])
        self.nt1 = self.nind1[-1]
        # total degree of freedom and parameter index for the second order
        self.nind2 = [0]
        self.pairs = []
        for d in range(self.D-1):
            for f in range(d+1, self.D):
                self.pairs.append((d,f))
                self.nind2.append(self.nind2[-1] + self.n[d]*self.n[f])
        self.nt2 = self.nind2[-1]
        # fidelity terms
        self.B = np.zeros((self.nt1+self.nt2, self.nt1+self.nt2))
        self.b = np.zeros(self.nt1+self.nt2)
        self.get_Bb()
        self.Bfull = np.triu(self.B, 1) + np.transpose(np.triu(self.B, 1)) + np.diag(np.diag(self.B))
        # gradient matrices
        self.K1 = [None] * self.D
        self.K2 = [None] * len(self.pairs)
        # get the weight type for every component
        self.w1 = np.zeros(self.D)
        self.w2 = np.zeros(len(self.pairs))

        # least norm least square solution
        self.uls = np.zeros(len(self.b))
        self.u1ls = [None] * self.D
        self.u2ls = [None] * len(self.pairs)
        self.output_Bb()

    def output_Bb(self, printb=False):
        gamma, u = np.linalg.eigh(self.B, UPLO='U')
        mask = (np.abs(gamma) > 1e-14)
        gamma = gamma[mask]
        u = u[:, mask]
        # check correctness of the B and b
        if abs(np.linalg.norm(self.b - np.dot(u, np.dot(u.T, self.b)))) > 1e-14 * np.linalg.norm(self.b):
            print "Warning: error in computing B and b"
        # least norm least square solution
        self.uls = np.dot(u / gamma, np.dot(u.T, self.b))
        # postprocessing
        for d in range(self.D):
            self.u1ls[d] = self.uls[self.nind1[d]:self.nind1[d + 1]]
        for pairid, pair in enumerate(self.pairs):
            u2flat = self.uls[self.nt1 + self.nind2[pairid]:self.nt1 + self.nind2[pairid + 1]]
            self.u2ls[pairid] = np.reshape(u2flat, (self.n[pair[0]], self.n[pair[1]]), order='F')
        # write Bb to txt file: change to the form of \|X u - y\|_2^2
        if printb:
            X = np.dot(u * np.sqrt(gamma), u.T)
            y = np.dot(u / np.sqrt(gamma), np.dot(u.T, self.b))
            data = pd.DataFrame(data=np.hstack((X, y[:, np.newaxis])))
            data.to_csv('Bb.csv')

    def get_Bb(self):
        '''
            # note that we only compute the upper triangular part of B
            B = [B11, B12; B21, B22] Only B11, B12, B22 are computed
        '''
        # the 0-th order term
        mg = self.m*self.g
        mgave = np.sum(mg)/self.msum

        # the first-order terms: B11
        for d in range(self.D):
            reduced = tuple(np.delete(range(self.D), d))
            gind = range(self.nind1[d], self.nind1[d+1])
            # compute amg
            self.b[gind] = np.sum(mg, axis=reduced)     # A_d^T M g
            # compute ama
            ama_temp = np.sum(self.m, axis=reduced)     # A_d^T M 1
            self.b[gind] -= mgave*ama_temp              # -mgave * A^T M 1
            # diagonal block
            self.B[gind, gind] = ama_temp       # we will use this quantity a few times, so not do the minus part right now
            # off diagonal block
            for f in range(d+1, self.D):
                ginf = range(self.nind1[f], self.nind1[f+1])
                reducedf = tuple(np.delete(range(self.D), [d, f]))
                self.B[np.ix_(gind, ginf)] = np.sum(self.m, axis=reducedf)      # A_d^T M A_f, no minus again!

        # the second order terms: B12 and B22
        for pairid, pair in enumerate(self.pairs):
            reducepair = tuple(np.delete(range(self.D), pair))
            gin0 = range(self.nind1[pair[0]], self.nind1[pair[0]+1])
            gin1 = range(self.nind1[pair[1]], self.nind1[pair[1]+1])
            ginpair = range(self.nt1+self.nind2[pairid], self.nt1+self.nind2[pairid+1])
            # compute amg
            self.b[ginpair] = np.sum(mg, axis=reducepair).flatten(order='F')  # A_(f,g)^T M g
            # compute ama
            ama_temp = self.B[np.ix_(gin0, gin1)]  # A_{f,g}^T M 1
            self.b[ginpair] -= mgave*ama_temp.flatten(order='F')        # -mgave * A^T M 1
            # diagonal block of B22
            self.B[ginpair, ginpair] = ama_temp.flatten(order='F')
            self.B[np.ix_(ginpair, ginpair)] -= np.outer(ama_temp.flatten(order='F'), ama_temp.flatten(order='F'))/self.msum
            # off diagonal block
            # cross between first and second terms: B12
            nstart = self.nt1 + self.nind2[pairid]
            for d in range(self.D):
                gind = range(self.nind1[d], self.nind1[d + 1])
                if d == pair[0]:
                    for i in range(self.n[d]):
                        self.B[self.nind1[d] + i, range(nstart + i, nstart + self.n[pair[0]]*self.n[pair[1]],
                                                      self.n[pair[0]])] = ama_temp[i, :]       # A_{f}^T M A_{f,g}
                elif d == pair[1]:
                    for i in range(self.n[d]):
                        self.B[self.nind1[d] + i, range(nstart + i*self.n[pair[0]], nstart + (i+1)*self.n[pair[0]])] \
                            = ama_temp[:, i]  # A_{g}^T M A_{f,g}
                else:
                    reducepaird = tuple(np.delete(range(self.D), list(pair)+[d]))
                    ama3_temp = np.sum(self.m, axis=reducepaird)        # A_{d}^T M A_{f,g}
                    if d < pair[0]:
                        for i in range(self.n[d]):
                            self.B[self.nind1[d] + i, ginpair] = (ama3_temp[i, :, :]).flatten(order='F')  # d < f < g
                    elif pair[0] < d < pair[1]:
                        for i in range(self.n[d]):
                            self.B[self.nind1[d] + i, ginpair] = (ama3_temp[:, i, :]).flatten(order='F')  # f < d < g
                    elif d > pair[1]:
                        for i in range(self.n[d]):
                            self.B[self.nind1[d] + i, ginpair] = (ama3_temp[:, :, i]).flatten(order='F')  # f < g < d
                    else:
                        print "Wrong at the first-second cross terms!"
                # do the minus
                self.B[np.ix_(gind, ginpair)] -= np.outer(self.B[gind, gind], ama_temp.flatten(order='F')) / self.msum
            # cross terms in the second order: B22
            for pairid2 in range(pairid+1, len(self.pairs)):
                pair2 = self.pairs[pairid2]
                ginpair2 = range(self.nt1 + self.nind2[pairid2], self.nt1 + self.nind2[pairid2 + 1])
                nstart2 = self.nt1 + self.nind2[pairid2]
                if pair2[0] == pair[0]:
                    # f = f', g < g', // f = f' < g < g'
                    if not (pair[1] < pair2[1]):
                        print "Wrong at f = f', g < g'!"
                    reducepair2 = tuple(np.delete(range(self.D), list(pair)+[pair2[1]]))
                    ama3_temp = np.sum(self.m, axis=reducepair2)  # A_{f,g}^T M A_{f,g'}
                    for i2 in range(self.n[pair2[0]]*self.n[pair2[1]]):
                        # note: we take the column-major index : order = 'F'
                        g2 = i2//self.n[pair2[0]]
                        f2 = i2 - g2*self.n[pair2[0]]
                        self.B[range(nstart+f2, nstart+self.n[pair[0]]*self.n[pair[1]],
                                     self.n[pair[0]]), nstart2+i2] = ama3_temp[f2, :, g2]
                elif pair2[0] == pair[1]:
                    # f' = g, f< g = f' < g'
                    if not (pair[0] < pair2[1]):
                        print "Wrong at f< g = f' < g'!"
                    reducepair2 = tuple(np.delete(range(self.D), list(pair)+[pair2[1]]))
                    ama3_temp = np.sum(self.m, axis=reducepair2)  # A_{f,g}^T M A_{g,g'}
                    for i2 in range(self.n[pair2[0]] * self.n[pair2[1]]):
                        # note: we take the column-major index : order = 'F'
                        g2 = i2 // self.n[pair2[0]]
                        f2 = i2 - g2 * self.n[pair2[0]]
                        self.B[range(nstart + f2*self.n[pair[0]], nstart + (f2+1) * self.n[pair[0]]), nstart2+i2] \
                            = ama3_temp[:, f2, g2]
                elif pair2[1] == pair[1]:
                    # g' = g, f< f' // f < f' < g = g'
                    if not (pair[0] < pair2[0]):
                        print "Wrong at f< f', g= g'!"
                    reducepair2 = tuple(np.delete(range(self.D), list(pair)+[pair2[0]]))
                    ama3_temp = np.sum(self.m, axis=reducepair2)  # A_{f,g}^T M A_{f',g}
                    for i2 in range(self.n[pair2[0]] * self.n[pair2[1]]):
                        # note: we take the column-major index : order = 'F'
                        g2 = i2 // self.n[pair2[0]]
                        f2 = i2 - g2 * self.n[pair2[0]]
                        self.B[range(nstart + g2 * self.n[pair[0]], nstart + (g2 + 1) * self.n[pair[0]]), nstart2+i2] \
                            = ama3_temp[:, f2, g2]
                elif set(pair).isdisjoint(set(pair2)):
                    # no same index
                    reducepair2 = tuple(np.delete(range(self.D), list(pair)+list(pair2)))
                    ama4_temp = np.sum(self.m, axis=reducepair2)        # A_{f,g}^T M A_{f',g'}
                    if not (pair[0] < pair2[0] < pair2[1]):
                        print "Wrong in the case: no index is the same"
                    if pair[0] < pair[1] < pair2[0]:
                        # f < g < f' < g'
                        for i2 in range(self.n[pair2[0]] * self.n[pair2[1]]):
                            # note: we take the column-major index : order = 'F'
                            g2 = i2 // self.n[pair2[0]]
                            f2 = i2 - g2 * self.n[pair2[0]]
                            self.B[ginpair, nstart2+i2] = (ama4_temp[:, :, f2, g2]).flatten(order='F')
                    elif pair2[0] < pair[1] < pair2[1]:
                        # f < f' < g < g'
                        for i2 in range(self.n[pair2[0]] * self.n[pair2[1]]):
                            # note: we take the column-major index : order = 'F'
                            g2 = i2 // self.n[pair2[0]]
                            f2 = i2 - g2 * self.n[pair2[0]]
                            self.B[ginpair, nstart2 + i2] = (ama4_temp[:, f2, :, g2]).flatten(order='F')
                    elif pair2[1] < pair[1]:
                        # f < f' < g' < g
                        for i2 in range(self.n[pair2[0]] * self.n[pair2[1]]):
                            # note: we take the column-major index : order = 'F'
                            g2 = i2 // self.n[pair2[0]]
                            f2 = i2 - g2 * self.n[pair2[0]]
                            self.B[ginpair, nstart2 + i2] = (ama4_temp[:, f2, g2, :]).flatten(order='F')
                    else:
                        print "Wrong in the case: no index is the same, f < f' < g', f< g do not hold!"
                else:
                    print "Wrong at branches in the cross terms in the second order"
        # minus avarage for the second-order cross terms
        for pairid, pair in enumerate(self.pairs):
            gin0 = range(self.nind1[pair[0]], self.nind1[pair[0]+1])
            gin1 = range(self.nind1[pair[1]], self.nind1[pair[1]+1])
            ginpair = range(self.nt1+self.nind2[pairid], self.nt1+self.nind2[pairid+1])
            ama_temp = self.B[np.ix_(gin0, gin1)]  # A_{f,g}^T M 1
            # cross terms in the second order
            for pairid2 in range(pairid + 1, len(self.pairs)):
                pair2 = self.pairs[pairid2]
                gin02 = range(self.nind1[pair2[0]], self.nind1[pair2[0] + 1])
                gin12 = range(self.nind1[pair2[1]], self.nind1[pair2[1] + 1])
                ginpair2 = range(self.nt1 + self.nind2[pairid2], self.nt1 + self.nind2[pairid2 + 1])
                ama_temp2 = self.B[np.ix_(gin02, gin12)]  # A_{f',g'}^T M 1
                self.B[np.ix_(ginpair, ginpair2)] -= np.outer(ama_temp.flatten(order='F'), ama_temp2.flatten(order='F'))/self.msum
        # minus avarage for the second-order cross terms
        for d in range(self.D):
            gind = range(self.nind1[d], self.nind1[d + 1])
            ama_temp = self.B[gind, gind]  # A_d^T M 1
            for f in range(d, self.D):
                ginf = range(self.nind1[f], self.nind1[f + 1])
                ama_temp2 = self.B[ginf, ginf]  # A_f^T M 1
                self.B[np.ix_(gind, ginf)] -= np.outer(ama_temp, ama_temp2)/self.msum

    #
    # def constructv(self, pair, i):
    #     # help to debug get_Bb. construct a unit tensor
    #     g = i // self.n[pair[0]]
    #     f = i - g * self.n[pair[0]]
    #     v = np.zeros(self.m.shape)
    #     if pair == (0,1):
    #         v[f,g,:,:] = 1
    #     elif pair == (0,2):
    #         v[f,:,g,:] = 1
    #     elif pair == (0,3):
    #         v[f,:,:,g] = 1
    #     elif pair == (1,2):
    #         v[:,f,g,:] = 1
    #     elif pair == (1,3):
    #         v[:,f,:,g] = 1
    #     elif pair == (2,3):
    #         v[:,:,f,g] = 1
    #     else:
    #         print "Something wrong in constructv!"
    #     return v
    #
    # def testB(self, pairid, pairid2):
    #     # test the computation of B for the second-order cross terms in the first three dimensions
    #     # (0,1) vs (0,2)
    #     pair = self.pairs[pairid]
    #     ginpair = range(self.nt1 + self.nind2[pairid], self.nt1 + self.nind2[pairid + 1])
    #     pair2 = self.pairs[pairid2]
    #     ginpair2 = range(self.nt1 + self.nind2[pairid2], self.nt1 + self.nind2[pairid2 + 1])
    #     Btest = self.B[np.ix_(ginpair, ginpair2)]
    #     B1 = np.zeros(Btest.shape)
    #     B2 = np.zeros(Btest.shape)
    #     # compute the corresponding tensors
    #     vpair = [self.constructv(pair, i) for i in range(self.n[pair[0]]*self.n[pair[1]])]
    #     vpair2 = [self.constructv(pair2, j) for j in range(self.n[pair2[0]]*self.n[pair2[1]])]
    #     # the most basic method
    #     for i in range(self.n[pair[0]]*self.n[pair[1]]):
    #         for j in range(self.n[pair2[0]]*self.n[pair2[1]]):
    #             B1[i,j] = np.sum(vpair[i]*self.m*vpair2[j])
    #     # a little more advanced
    #     reducepair = tuple(np.delete(range(self.D), pair))
    #     for j in range(self.n[pair2[0]]*self.n[pair2[1]]):
    #         mg = self.m * vpair2[j]
    #         temp = np.sum(mg, axis=reducepair)
    #         B2[:,j] = temp.flatten(order='F')
    #
    #     return Btest, B1, B2
    #
    # def testB2(self):
    #     for pairid, pair in enumerate(self.pairs):
    #         for pairid2 in range(pairid+1, len(self.pairs)):
    #             Btest, B1, B2 = self.testB(pairid, pairid2)
    #             if np.linalg.norm(Btest-B1) + np.linalg.norm(Btest-B2) < 1e-14:
    #                 print "Works good for pairid, pairid2:", pairid, pairid2
    #             else:
    #                 print "Works bad for pairid, pairid2:", pairid, pairid2
    #                 print "Btest-B1", np.linalg.norm(Btest-B1)
    #                 print "Btest-B2", np.linalg.norm(Btest-B2)d

    def set_graph(self, d, A):
        # A is the adjacency matrix of feature d
        # make sure that the node index in A is consistent with that in g and m
        if type(A).__module__ == np.__name__:
            self.A[d] = np.ones((self.n[d], self.n[d] + 1), dtype=bool)
            self.A[d][:, 0:self.n[d]] = A
        elif A == "full":
            self.A[d] = np.ones((self.n[d], self.n[d]+1), dtype=bool)
            self.A[d][range(0, self.n[d]), range(0, self.n[d])] = False
        elif A == "linear":
            self.A[d] = np.zeros((self.n[d], self.n[d]+1), dtype=bool)
            self.A[d][:, self.n[d]] = True
            self.A[d][range(1, self.n[d]), range(0, self.n[d] - 1)] = True
            self.A[d][range(0, self.n[d] - 1), range(1, self.n[d])] = True
        elif A == "circular":
            self.A[d] = np.zeros((self.n[d], self.n[d]+1), dtype=bool)
            self.A[d][:, self.n[d]] = True
            self.A[d][range(1, self.n[d]), range(0, self.n[d] - 1)] = True
            self.A[d][range(0, self.n[d] - 1), range(1, self.n[d])] = True
            self.A[d][0, self.n[d]-1] = True
            self.A[d][self.n[d]-1, 0] = True
        else:
            print("Unrecognized adjacency matrix, please input again!")

        self.nlevel[d] = np.sum(self.A[d], axis=1)

    def get_K(self):
        # K for first order effect
        for d in range(self.D):
            self.K1[d] = np.zeros((np.sum(self.A[d]), self.n[d]))
            edgestart = 0
            for node in range(self.n[d]):
                edgeid = range(edgestart, edgestart+np.sum(self.A[d][node, 0:self.n[d]]))
                self.K1[d][edgeid, self.A[d][node, 0:self.n[d]]] = 1.0/self.nlevel[d][node]
                self.K1[d][edgeid, node] = -1.0 / self.nlevel[d][node]
                # the L1 penalty // the weight is the same with other edges
                self.K1[d][edgeid[-1]+1, node] = -1.0 / self.nlevel[d][node]
                # updage edgestart
                edgestart = edgeid[-1] + 2
        # K for second order effect
        for pairid, pair in enumerate(self.pairs):
            T0, n0 = self.K1[pair[0]].shape
            T1, n1 = self.K1[pair[1]].shape
            if not (n0 == self.n[pair[0]] and n1 == self.n[pair[1]]):
                print "Error in the K for second order effect!"
            self.K2[pairid] = np.zeros((n1*T0 + n0*T1, n0*n1))
            for i in range(n1):
                self.K2[pairid][i*T0:(i+1)*T0, i*n0:(i+1)*n0] = self.K1[pair[0]]
            for j in range(n0):
                self.K2[pairid][(j*T1+n1*T0):((j+1)*T1+n1*T0), j::n0] = self.K1[pair[1]]

    def set_w(self, Ns=100, printk=False):
        try:
            self.w1 = pd.read_csv('w1.csv', index_col=0).values
            self.w2 = pd.read_csv('w2.csv', index_col=0).values
        except Exception, e:
            print "Error in reading:", 'w1.csv', 'w2.csv'
            print e
            # compute the weight by linear programming: w_i = E_g[ max <A_i^T g_star, u_i>  s.t. \|K_i u_i\|_1 \le 1 ]
            samples1 = np.zeros((Ns, self.D))
            samples2 = np.zeros((Ns, len(self.pairs)))
            for t in range(Ns):
                print "sample:", t
                # sample g ~ N(0, M^{-1}) and compute g_star
                g = np.random.normal(loc=0.0, scale=1.0, size=self.m.shape) / np.sqrt(self.m)
                mg = self.m * g
                mgave = np.sum(mg) / self.msum
                g_star = self.m * (g - mgave)
                # first order
                for d in range(self.D):
                    reduced = tuple(np.delete(range(self.D), d))
                    btemp = np.sum(g_star, axis=reduced)
                    samples1[t, d] = -cvxopt_solve_minmax(-btemp, self.K1[d], solver='glpk')
                # second order
                legend2 = []
                for pairid, pair in enumerate(self.pairs):
                    reducepair = tuple(np.delete(range(self.D), pair))
                    btemp = np.sum(g_star, axis=reducepair)
                    samples2[t, pairid] = -cvxopt_solve_minmax(-btemp.flatten(order='F'), self.K2[pairid], solver='glpk')
            # save samples
            legend1 = ["X%d" % (d+1) for d in range(self.D)]
            data1 = pd.DataFrame(data=samples1, columns=legend1)
            data1.to_csv('data1.csv')
            legend2 = ["X%d-X%d" % (pair[0] + 1, pair[1] + 1) for pair in self.pairs]
            data2 = pd.DataFrame(data=samples2, columns=legend2)
            data2.to_csv('data2.csv')
            # compute the average
            self.w1 = np.mean(samples1, axis=0)
            w1 = pd.DataFrame(data=self.w1, index=legend1)
            w1.to_csv('w1.csv')
            self.w2 = np.mean(samples2, axis=0)
            w2 = pd.DataFrame(data=self.w2, index=legend2)
            w2.to_csv('w2.csv')

        # multiply the weight for each term
        for d in range(self.D):
            self.K1[d] *= self.w1[d]
        for pairid, pair in enumerate(self.pairs):
            self.K2[pairid] *= self.w2[pairid]
        # output to csv file
        if printk:
            Ktotal = self.K1 + self.K2
            shapes = np.array([K.shape for K in Ktotal])
            data = np.zeros(np.sum(shapes, axis=0), dtype=Ktotal[0].dtype)
            r, c = 0, 0
            for i, (rr, cc) in enumerate(shapes):
                data[r:r + rr, c:c + cc] = Ktotal[i]
                r += rr
                c += cc
            data = pd.DataFrame(data=data)
            data.to_csv('K.csv')

    def apply_F(self, Y1, Y2):
        tv = 0.
        for d in range(self.D):
            tv += np.sum(np.abs(Y1[d]))
        for pairid, pair in enumerate(self.pairs):
            tv += np.sum(np.abs(Y2[pairid]))
        return tv

    def prox_Fstar(self, Y1, Y2):
        # since this is the proxy of a delta function, the step size sigma does not matter
        for d in range(self.D):
            # project every item to interval [-1,1]
            Y1[d] = map(lambda x: x/max(1, abs(x)), Y1[d])
        for pairid, pair in enumerate(self.pairs):
            # project every item to interval [-1,1]
            Y2[pairid] = map(lambda x: x / max(1, abs(x)), Y2[pairid])

    def apply_K(self, x, Y1, Y2):
        # x is the state variable, Y is the flux variable to store the result.
        # Y has the same dimension as A
        for d in range(self.D):
            xd = x[self.nind1[d]:self.nind1[d+1]]
            Y1[d] = np.dot(self.K1[d], xd)
        for pairid, pair in enumerate(self.pairs):
            xpair = x[self.nt1+self.nind2[pairid]:self.nt1+self.nind2[pairid+1]]
            Y2[pairid] = np.dot(self.K2[pairid], xpair)

    def apply_Kstar(self, x, Y1, Y2):
        for d in range(self.D):
            x[self.nind1[d]:self.nind1[d+1]] = np.dot(Y1[d], self.K1[d])
        for pairid, pair in enumerate(self.pairs):
            x[self.nt1 + self.nind2[pairid]:self.nt1 + self.nind2[pairid + 1]] = np.dot(Y2[pairid], self.K2[pairid])

    def get_tinv_sigmainv(self):
        tinv = np.zeros(self.nt1+self.nt2)
        sigmainv1 = [None]*self.D
        sigmainv2 = [None]*len(self.pairs)
        # alpha = 1
        for d in range(self.D):
            tinv[self.nind1[d]:self.nind1[d + 1]] = np.sum(np.abs(self.K1[d]), axis=0)
            sigmainv1[d] = np.sum(np.abs(self.K1[d]), axis=1)
        for pairid, pair in enumerate(self.pairs):
            tinv[self.nt1 + self.nind2[pairid]:self.nt1 + self.nind2[pairid + 1]] = np.sum(np.abs(self.K2[pairid]), axis=0)
            sigmainv2[pairid] = np.sum(np.abs(self.K2[pairid]), axis=1)

        return tinv, sigmainv1, sigmainv2

    def P_PDALG1(self, Lambda, n_it, return_energy=True):
        '''
            # Lambda is the weight for the accuracy term
            Preconditioned Chambolle-Pock algorithm for the minimization of the objective function
                \|K x\|_1 + 1/2*Lambda*||B*x - b||_2^2
            Lambda : weight of the TV penalization (the higher Lambda, the more sparse is the solution)
            n_it : number of iterations
            return_energy: if True, an array containing the values of the objective function will be returned
        '''
        theta = 1.0
        tinv, sigmainv1, sigmainv2 = self.get_tinv_sigmainv()
        self.Gamma, self.U = np.linalg.eigh(Lambda*self.B + np.diag(tinv), UPLO='U')

        # primal dual splitting method
        # initialize
        # primal variable
        x = np.zeros(self.nt1+self.nt2)
        x_tilde = 1.0*x
        # dual variables
        Y1 = [None]*self.D
        Y1_tilde = [None]*self.D
        for d in range(self.D):
            m, n = self.K1[d].shape
            Y1[d] = np.zeros(m)
            Y1_tilde[d] = np.zeros(m)
        Y2 = [None] * len(self.pairs)
        Y2_tilde = [None] * len(self.pairs)
        for pairid, pair in enumerate(self.pairs):
            m, n = self.K2[pairid].shape
            Y2[pairid] = np.zeros(m)
            Y2_tilde[pairid] = np.zeros(m)

        if return_energy:
            en = np.zeros(n_it)
        for k in range(n_it):
            # update dual variables
            self.apply_K(x_tilde, Y1_tilde, Y2_tilde)
            for d in range(self.D):
                Y1[d] += Y1_tilde[d]/sigmainv1[d]
            for pairid, pair in enumerate(self.pairs):
                Y2[pairid] += Y2_tilde[pairid]/sigmainv2[pairid]
            self.prox_Fstar(Y1, Y2)
            # update primal variable
            x_old = 1.0*x
            self.apply_Kstar(x_tilde, Y1, Y2)
            x = Lambda*self.b + x_old*tinv - x_tilde
            # self.prox_G(x, tau)
            x = np.dot(self.U/self.Gamma, np.dot(np.transpose(self.U), x))
            x_tilde = x + theta*(x - x_old)
            # calculate norms
            if return_energy:
                # fidelity: no \|g\|^2 term
                fidelity = 0.5*np.dot(x, np.dot(self.Bfull, x)) - np.dot(x, self.b)
                # TV norm
                self.apply_K(x, Y1_tilde, Y2_tilde)
                tv = self.apply_F(Y1_tilde, Y2_tilde)
                # energy
                energy = Lambda*fidelity + tv
                en[k] = energy
                if k % 10 == 0:
                    print("[%d] : energy %1.10e \t fidelity %e \t TV %e" % (k, energy, fidelity, tv))

        # postprocessing
        u1 = [None]*self.D
        for d in range(self.D):
            u1[d] = x[self.nind1[d]:self.nind1[d + 1]]
        u2 = [None] * len(self.pairs)
        for pairid, pair in enumerate(self.pairs):
            u2flat = x[self.nt1 + self.nind2[pairid]:self.nt1 + self.nind2[pairid + 1]]
            u2[pairid] = np.reshape(u2flat, (self.n[pair[0]], self.n[pair[1]]), order='F')

        # return value
        if return_energy:
            return en, u1, u2
        else:
            return u1, u2


def cvxopt_solve_minmax(b, S, solver='glpk'):
    # solve min <b, u>  subject to: \|S u\|_1 \le 1
    # dimensions
    m, n = S.shape
    # cvxopt objective format: c.T x
    c = np.hstack([b, np.zeros(m)])

    # cvxopt constraint format: G * x - h <= 0
    # S u \le v , -S u \le v, 1^T v \le 1
    G = np.vstack([
        np.hstack([S, -np.eye(m)]),
        np.hstack([-S, -np.eye(m)]),
        np.hstack([np.zeros(n), np.ones(m)])])
    h = np.hstack([np.zeros(2*m), [1]])

    c = cvxopt.matrix(c)
    G = cvxopt.matrix(G)
    # G = cvxopt.sparse(G)
    h = cvxopt.matrix(h)
    sol = cvxopt.solvers.lp(c, G, h, solver=solver)
    if not(sol['status'] == 'optimal'):
        print "Something wrong in the linear programming"
    return sol['primal objective']