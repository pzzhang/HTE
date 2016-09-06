import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxopt
from grouping import grouping
from sklearn import linear_model
# disable GLPK output
cvxopt.solvers.options['LPX_K_MSGLEV'] = 0         # old versions of cvxopt
cvxopt.solvers.options['msg_lev'] = 'GLP_MSG_OFF'  # works on cvxopt 1.1.7


class TVreg1st(object):
    def __init__(self, g, m):
        # g and m are D-dimensional arrays with the same shape
        # g is the point-wise averaged noisy measurement
        # m is the effective percentage of measurements at the corresponding point (inverse of variance)
        #   min_u \|\grad u\|_{1} + Lambda/2 \|u - g\|_{2,m}^2
        self.g = g
        self.n = np.array(g.shape)
        self.D = len(self.n)
        self.N = np.prod(self.n)
        self.m = m/(1.0*self.N)         #normalization, it is not necessary to do this.
        self.msum = np.sum(self.m)

        # total degree of freedom and parameter index for 1st order method
        self.nind1 = [0]
        for d in range(self.D):
            self.nind1.append(self.nind1[-1] + self.n[d])
        self.nt1 = self.nind1[-1]
        self.nT = self.nind1[-1]
        self.nind = list(self.nind1)
        # fidelity terms
        self.B = np.zeros((self.nT, self.nT))
        self.b = np.zeros(self.nT)
        self.c = 0
        self.get_Bbc()
        # the adjacency matrix A and the degree (on each level) at each node nlevel
        # default: fully connected
        self.A1 = [np.ones((self.n[d], self.n[d]+1), dtype=bool) for d in range(self.D)]
        for d in range(self.D):
            self.A1[d][range(0, self.n[d]), range(0, self.n[d])] = False
        self.nlevel = [np.sum(self.A1[d], axis=1) for d in range(self.D)]
        # gradient matrices and second order adjacency matrices based on tensor-product topology
        self.K1 = [None] * self.D
        self.K = [None] * self.D
        self.A = [None] * self.D
        # get the weight type for every component
        self.w1 = np.zeros(self.D)
        self.w = np.zeros(self.D)
        self.w1_lasso = np.zeros(self.D)
        self.w_lasso = np.zeros(self.D)

        # least norm least square solution
        self.uls = np.zeros(len(self.b))
        self.u1ls = [None] * self.D
        self.X = np.zeros((len(self.b), len(self.b)))
        self.y = np.zeros(len(self.b))
        self.output_Bb()

    def output_Bb(self, printb=False):
        gamma, u = np.linalg.eigh(self.B, UPLO='U')
        mask = (np.abs(gamma) > np.max(gamma)*1e-14)
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
        # write Bb to txt file: change to the form of \|X u - y\|_2^2
        self.X = np.dot(u * np.sqrt(gamma), u.T)
        self.y = np.dot(u / np.sqrt(gamma), np.dot(u.T, self.b))
        if printb:
            data = pd.DataFrame(data=np.hstack((self.X, self.y[:, np.newaxis])))
            data.to_csv('Bb.csv')

    def get_Bbc(self):
        '''
            # note that we only compute the upper triangular part of B
            B = [B11, B12; B21, B22] Only B11, B12, B22 are computed
        '''
        # the 0-th order term
        mg = self.m*self.g
        mgave = np.sum(mg)/self.msum
        self.c = np.sum(self.g * mg) - self.msum * mgave**2

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

        # minus avarage for the first-order cross terms
        for d in range(self.D):
            gind = range(self.nind1[d], self.nind1[d + 1])
            ama_temp = self.B[gind, gind]  # A_d^T M 1
            for f in range(d, self.D):
                ginf = range(self.nind1[f], self.nind1[f + 1])
                ama_temp2 = self.B[ginf, ginf]  # A_f^T M 1
                self.B[np.ix_(gind, ginf)] -= np.outer(ama_temp, ama_temp2)/self.msum

        # B is symmetric and just now we only compute its upper triangular part
        self.B = np.triu(self.B, 1) + np.transpose(np.triu(self.B, 1)) + np.diag(np.diag(self.B))

    def set_graph(self, d, A):
        # A is the adjacency matrix of feature d
        # make sure that the node index in A is consistent with that in g and m
        if type(A).__module__ == np.__name__:
            self.A1[d] = np.ones((self.n[d], self.n[d] + 1), dtype=bool)
            self.A1[d][:, 0:self.n[d]] = A
        elif A == "full":
            self.A1[d] = np.ones((self.n[d], self.n[d]+1), dtype=bool)
            self.A1[d][range(0, self.n[d]), range(0, self.n[d])] = False
        elif A == "linear":
            self.A1[d] = np.zeros((self.n[d], self.n[d]+1), dtype=bool)
            self.A1[d][:, self.n[d]] = True
            self.A1[d][range(1, self.n[d]), range(0, self.n[d] - 1)] = True
            self.A1[d][range(0, self.n[d] - 1), range(1, self.n[d])] = True
        elif A == "circular":
            self.A1[d] = np.zeros((self.n[d], self.n[d]+1), dtype=bool)
            self.A1[d][:, self.n[d]] = True
            self.A1[d][range(1, self.n[d]), range(0, self.n[d] - 1)] = True
            self.A1[d][range(0, self.n[d] - 1), range(1, self.n[d])] = True
            self.A1[d][0, self.n[d]-1] = True
            self.A1[d][self.n[d]-1, 0] = True
        else:
            print("Unrecognized adjacency matrix, please input again!")

        self.nlevel[d] = np.sum(self.A1[d], axis=1)

    def get_K(self):
        # K for first order effect
        for d in range(self.D):
            self.K1[d] = np.zeros((np.sum(self.A1[d]), self.n[d]))
            edgestart = 0
            for node in range(self.n[d]):
                edgeid = range(edgestart, edgestart+np.sum(self.A1[d][node, 0:self.n[d]]))
                self.K1[d][edgeid, self.A1[d][node, 0:self.n[d]]] = 1.0/self.nlevel[d][node]
                self.K1[d][edgeid, node] = -1.0 / self.nlevel[d][node]
                # the L1 penalty // the weight is the same with other edges
                self.K1[d][edgeid[-1]+1, node] = -1.0 / self.nlevel[d][node]
                # updage edgestart
                edgestart = edgeid[-1] + 2
        # total K
        self.K = [1.0*k for k in self.K1]
        self.A = [1*a for a in self.A1]

    def set_w(self, Ns=100, printk=False):
        try:
            self.w1 = pd.read_csv('w1.csv', index_col=0).values.flatten()
        except Exception, e:
            print "Error in reading:", 'w1.csv'
            print e
            # compute the weight by linear programming: w_i = E_g[ max <A_i^T g_star, u_i>  s.t. \|K_i u_i\|_1 \le 1 ]
            samples1 = np.zeros((Ns, self.D))
            for t in range(Ns):
                print "sample:", t
                # sample g ~ N(0, M) and compute g_star
                g = np.random.normal(loc=0.0, scale=1.0, size=self.m.shape) * np.sqrt(self.m)
                gave = np.sum(g) / self.msum
                g_star = g - self.m * gave
                # first order
                for d in range(self.D):
                    reduced = tuple(np.delete(range(self.D), d))
                    btemp = np.sum(g_star, axis=reduced)
                    samples1[t, d] = -cvxopt_solve_minmax(-btemp, self.K1[d], solver='glpk')
            # save samples
            legend1 = ["X%d" % (d+1) for d in range(self.D)]
            data1 = pd.DataFrame(data=samples1, columns=legend1)
            data1.to_csv('data1.csv')
            # compute the average
            self.w1 = np.mean(samples1, axis=0)
            w1 = pd.DataFrame(data=self.w1, index=legend1)
            w1.to_csv('w1.csv')

        # multiply the weight for each term: note that self.K1 and self.K2 are not changed!
        self.w = 1.0*self.w1
        for i, K in enumerate(self.K):
            K *= self.w[i]
        # output to csv file
        if printk:
            data = to_blockdiag(self.K)
            data = pd.DataFrame(data=data)
            data.to_csv('K.csv')

    def apply_F(self, Y):
        tv = 0.
        for y in Y:
            tv += np.sum(np.abs(y))
        return tv

    def prox_Fstar(self, Y):
        # since this is the proxy of a delta function, the step size sigma does not matter
        for i, y in enumerate(Y):
            # project every item to interval [-1,1]
            Y[i] = map(lambda x: x/max(1, abs(x)), y)

    def apply_K(self, x, Y):
        # x is the state variable, Y is the flux variable to store the result.
        for i, K in enumerate(self.K):
            xd = x[self.nind1[i]:self.nind1[i + 1]]
            Y[i] = np.dot(K, xd)

    def apply_Kstar(self, x, Y):
        for i, K in enumerate(self.K):
            x[self.nind1[i]:self.nind1[i + 1]] = np.dot(Y[i], K)

    def get_tinv_sigmainv(self):
        tinv = np.zeros(self.nT)
        sigmainv = [None]*len(self.K)
        # alpha = 1
        for i, K in enumerate(self.K):
            tinv[self.nind1[i]:self.nind1[i + 1]] = np.sum(np.abs(K), axis=0)
            sigmainv[i] = np.sum(np.abs(K), axis=1)

        return tinv, sigmainv

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
        tinv, sigmainv = self.get_tinv_sigmainv()
        Gamma, U = np.linalg.eigh(Lambda*self.B + np.diag(tinv), UPLO='U')

        # primal dual splitting method
        # initialize
        # primal variable
        x = np.zeros(self.nT)
        x_tilde = 1.0*x
        # dual variables
        Y = [None]*self.D
        Y_tilde = [None]*len(Y)
        for d in range(len(Y)):
            m, n = self.K[d].shape
            Y[d] = np.zeros(m)
            Y_tilde[d] = np.zeros(m)

        if return_energy:
            en = np.zeros(n_it)
        for k in range(n_it):
            # update dual variables
            self.apply_K(x_tilde, Y_tilde)
            for i, y_tilde in enumerate(Y_tilde):
                Y[i] += y_tilde/sigmainv[i]
            self.prox_Fstar(Y)
            # update primal variable
            x_old = 1.0*x
            self.apply_Kstar(x_tilde, Y)
            x = Lambda*self.b + x_old*tinv - x_tilde
            # self.prox_G(x, tau)
            x = np.dot(U/Gamma, np.dot(np.transpose(U), x))
            x_tilde = x + theta*(x - x_old)
            # calculate norms
            if return_energy:
                # fidelity: no \|g\|^2 term
                fidelity = 0.5*np.dot(x, np.dot(self.B, x)) - np.dot(x, self.b) + 0.5*self.c
                # TV norm
                self.apply_K(x, Y_tilde)
                tv = self.apply_F(Y_tilde)
                # energy
                energy = Lambda*fidelity + tv
                en[k] = energy
                if k % 10 == 0:
                    print("[%d] : energy %1.10e \t fidelity %e \t TV %e" % (k, energy, fidelity, tv))

        # postprocessing
        u1 = [None]*self.D
        for d in range(self.D):
            u1[d] = x[self.nind1[d]:self.nind1[d + 1]]

        # return value
        if return_energy:
            return en, u1
        else:
            return u1

    def P_PDALG2(self, lam, b, Gamma, U, tinv, sigmainv, theta, x, Y, tol=1e-8, max_iter=30000):
        # creat auxillary variables
        x_tilde = 1.0*x
        # dual variables
        Y_tilde = [np.array(y) for y in Y]
        # energy monitoring
        en = np.zeros(max_iter//100)
        # run pre-conditioned primal dual algorithm
        for k in range(max_iter):
            # update dual variables
            self.apply_K(x_tilde, Y_tilde)
            for i, y_tilde in enumerate(Y_tilde):
                Y[i] += y_tilde / sigmainv[i]
            self.prox_Fstar(Y)
            # update primal variable
            x_old = 1.0 * x
            self.apply_Kstar(x_tilde, Y)
            x = lam * b + x_old * tinv - x_tilde
            # self.prox_G(x, tau)
            x = np.dot(U / Gamma, np.dot(np.transpose(U), x))
            x_tilde = x + theta * (x - x_old)
            if k % 100 == 0:
                # energyid
                energyid = k/100
                # fidelity: no \|g\|^2 term
                fidelity = 0.5 * np.dot(x, np.dot(self.B, x)) - np.dot(x, b) + 0.5 * self.c
                # TV norm
                self.apply_K(x, Y_tilde)
                tv = self.apply_F(Y_tilde)
                # energy
                energy = lam * fidelity + tv
                en[energyid] = energy
                # check the stopping criteria
                if energyid > 5 and abs(en[energyid-1]-en[energyid]) < tol*(en[0]-en[1]):
                    break
        return k, x, Y

    def lasso_path(self, lam_min, lam_max, nstep=100, tol=1e-8):
        # the list of lambdas we will compute
        lam_list = 1.0/np.linspace(1.0/lam_min, 1.0/lam_max, nstep)
        # settings of the P_PDALG
        theta = 1.0
        tinv, sigmainv = self.get_tinv_sigmainv()
        # initial guess for the very beginning
        x = np.zeros(self.nT)
        Y = [None] * (self.D)
        for d in range(len(Y)):
            m, n = self.K[d].shape
            Y[d] = np.zeros(m)

        # compute lasso path
        u_list = [None]*nstep
        u_list[0] = 1.0*x
        noise_energy_list = [0.5*self.c]
        uls_list = [1.0*x]
        for i, lam in enumerate(lam_list):
            # preconditioning
            Gamma, U = np.linalg.eigh(lam * self.B + np.diag(tinv), UPLO='U')
            # store last step solution
            x_old = 1.0 * x
            num_iter, x, Y = self.P_PDALG2(lam, self.b, Gamma, U, tinv, sigmainv, theta, x_old, Y, tol=1e-8, max_iter=30000)
            u_list[i] = 1.0 * x
            # if change, do something
            is_changed = ((np.abs(x) > 2 * tol) == (np.abs(x_old) > 2 * tol))
            if not(is_changed.all()):
                # support changes
                x_ls, num_groups, noise_energy = self.blockLS(x, tol=tol)
                print "Case: lambdaId, Lamda = ", i, lam
                print "Number of non-zero groups = ", num_groups
                print "num_iter, Noise_Energy", num_iter, noise_energy
                self.print_effect(x_ls, lam, i, "LassoPath")
                uls_list.append(1.0*x_ls)
                noise_energy_list.append(noise_energy)

        return u_list, noise_energy_list, uls_list

    def bregman_iteration(self, lam, nstep = 100, tol=1e-8):
        # settings of the P_PDALG
        theta = 1.0
        tinv, sigmainv = self.get_tinv_sigmainv()
        # initial guess for the very beginning
        x = np.zeros(self.nT)
        Y = [None] * (self.D)
        for d in range(len(Y)):
            m, n = self.K[d].shape
            Y[d] = np.zeros(m)
        # preconditioning
        Gamma, U = np.linalg.eigh(lam * self.B + np.diag(tinv), UPLO='U')
        # compute the iteration
        u_list = [None]*nstep
        u_list[0] = 1.0*x
        noise_energy_list = [0.5*self.c]
        uls_list = [1.0*x]
        b = 0.0*self.b
        for i in range(nstep):
            # store last step solution
            x_old = 1.0 * x
            # update b
            b += self.b - np.dot(self.B, x_old)
            num_iter, x, Y = self.P_PDALG2(lam, b, Gamma, U, tinv, sigmainv, theta, x_old, Y, tol=1e-8, max_iter=30000)
            u_list[i] = 1.0 * x
            is_changed = ((np.abs(x) > 2*tol) == (np.abs(x_old) > 2*tol))
            if not(is_changed.all()):
                # support changes
                x_ls, num_groups, noise_energy = self.blockLS(x, tol=tol)
                print "Bregman iteration step = ", i
                print "Number of non-zero groups = ", num_groups
                print "num_iter, Noise_Energy", num_iter, noise_energy
                self.print_effect(x_ls, lam, i, "BregmanIteration")
                uls_list.append(1.0*x_ls)
                noise_energy_list.append(noise_energy)

        return u_list, noise_energy_list, uls_list

    def blockLS(self, x, tol):
        utrans = []
        for termid, termA in enumerate(self.A):
            xloc = x[self.nind[termid]:self.nind[termid+1]]
            uloc = grouping(xloc, termA, tol)
            utrans.append(uloc)
        # form the block diagonal matrix
        U = to_blockdiag(utrans)
        gamma, v = np.linalg.eigh(np.dot(U.T, np.dot(self.B, U)), UPLO='U')
        mask = (np.abs(gamma) > np.max(gamma)*1e-14)
        gamma = gamma[mask]
        v = v[:, mask]
        Ub = np.dot(self.b.T, U)
        a = np.dot(v / gamma, np.dot(v.T, Ub))
        uls = np.dot(U, a)
        noise_energy = -0.5*np.dot(Ub, a) + 0.5*self.c
        return uls, len(a), noise_energy

    def print_effect(self, u, lam, stepid, str_method):
        # get labels
        labels = []
        for d in range(self.D):
            labeld = ["X%d:1" % (d + 1)] + [str(i + 1) for i in range(1, self.n[d])]
            labels += labeld
        # plot computed result
        fig, ax = plt.subplots()
        plt.plot(u, '-*')
        plt.xlim(0, len(u))
        ax.set_xticks(range(len(u)))
        ax.set_xticklabels(labels, minor=False)
        plt.title("TV-regression-1st: Lambda%1.2e" % lam)
        # plt.show()
        plt.savefig('figure/firstorder/'+str_method+'/Step%d_Lambda%1.2e.png' % (stepid, lam))
        plt.close(fig)


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


def to_blockdiag(K):
    shapes = np.array([k.shape for k in K])
    data = np.zeros(np.sum(shapes, axis=0), dtype=K[0].dtype)
    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes):
        data[r:r + rr, c:c + cc] = K[i]
        r += rr
        c += cc
    return data