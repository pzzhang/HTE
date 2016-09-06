import numpy as np


class Chambolle(object):
    def __init__(self, g, m):
        # g and m are D-dimensional arrays with the same shape
        # g is the point-wise averaged noisy measurement
        # m is the effective percentage of measurements at the corresponding point
        #   min_f \|\grad f\|_{1} + Lambda/2 \|f - g\|_{2,m}^2
        self.g = g
        self.n = np.array(g.shape)
        # w is the weight in front of edge e(d,k,k')
        self.D = len(self.n)
        self.N = np.prod(self.n)
        self.m = m/(1.0*self.N)

        # the adjacency matrix and the degree (on each level) at each node nlevel
        # default: fully connected
        self.A = [np.ones((self.n[d], self.n[d]), dtype=bool) for d in range(self.D)]
        for d in range(self.D):
            self.A[d][range(0, self.n[d]), range(0, self.n[d])] = False
        self.nlevel = [(self.n[d]-1)*np.ones(self.n[d]) for d in range(self.D)]
        # get the weight for every edge
        self.w = [None]*self.D
        self.set_w(w_type=3)

        # attributes for the first-order additive model
        # total degree of freedom for 1st order method
        self.nt1 = sum(self.n)+1
        # parameter index for each parameter
        self.nind1 = [1]
        for d in range(self.D):
            self.nind1.append(self.nind1[-1] + self.n[d])
        # ama and amg
        self.ama0 = np.zeros((self.nt1, self.nt1))
        self.mu = np.zeros(self.D)
        self.ama = np.zeros((self.nt1, self.nt1))
        self.Gamma = 0.0
        self.U = 0.0
        self.amg = np.zeros(self.nt1)
        self.lssol = np.zeros(self.nt1)
        self.get_ama_amg()

    def get_ama_amg(self):
        mg = self.m*self.g
        # the constant term
        self.amg[0] = np.sum(mg)
        self.ama0[0,0] = np.sum(self.m)
        # the first-order terms
        for d in range(self.D):
            reduceD = tuple(range(d)+range(d+1, self.D))
            gind = range(self.nind1[d], self.nind1[d+1])
            # compute amg
            self.amg[gind] = np.sum(mg, axis=reduceD)
            # compute ama
            ama_temp = np.sum(self.m, axis=reduceD)
            # diagonal block is diagonal + ones-matrix
            self.ama0[gind, gind] = ama_temp
            # off diagonal block
            # get the 0-1 effect
            self.ama0[0, gind] = ama_temp
            self.ama0[gind, 0] = ama_temp
            # get the 1-1 effect
            for dprime in range(d+1,self.D):
                gindprime = range(self.nind1[dprime], self.nind1[dprime+1])
                reduceDprime = tuple(range(d) + range(d+1, dprime) + range(dprime+1, self.D))
                self.ama0[np.ix_(gind, gindprime)] = np.sum(self.m, axis=reduceDprime)
                self.ama0[np.ix_(gindprime, gind)] = np.transpose(self.ama0[np.ix_(gind, gindprime)])
        # get ama
        self.ama = 1.0*self.ama0
        for d in range(self.D):
            gind = range(self.nind1[d], self.nind1[d + 1])
            ama_temp = self.ama0[0, gind]
            # we take mu_d = np.mean(ama_temp)
            self.mu[d] = np.mean(ama_temp)
            self.ama[np.ix_(gind, gind)] += self.mu[d]*np.ones((self.n[d], self.n[d]))

        # eigen decomposition of ama = U * Gamma * U^T
        self.Gamma, self.U = np.linalg.eigh(self.ama)
        self.lssol = np.dot(self.U/self.Gamma, np.dot(np.transpose(self.U), self.amg))

    # def set_graph(self, A):
    #     # A is the adjacency matrix of each feature, A is a list that consists D boolean numpy matrices
    #     # make sure that the node index in A is consistent with that in g and m
    #     self.A = A
    #     for d in range(self.D):
    #         self.nlevel[d] = np.sum(self.A[d], axis=1)

    def set_graph(self, d, A):
        # A is the adjacency matrix of feature d
        # make sure that the node index in A is consistent with that in g and m
        if type(A).__module__ == np.__name__:
            self.A[d] = A
        elif A == "linear":
            self.A[d] = np.zeros((self.n[d], self.n[d]), dtype=bool)
            self.A[d][range(1, self.n[d]), range(0, self.n[d] - 1)] = True
            self.A[d][range(0, self.n[d] - 1), range(1, self.n[d])] = True
        elif A == "circular":
            self.A[d] = np.zeros((self.n[d], self.n[d]), dtype=bool)
            self.A[d][range(1, self.n[d]), range(0, self.n[d] - 1)] = True
            self.A[d][range(0, self.n[d] - 1), range(1, self.n[d])] = True
            self.A[d][0, -1] = True
            self.A[d][-1, 0] = True
        else:
            print("Unrecognizable adjacency matrix, please input again!")

        self.nlevel[d] = np.sum(self.A[d], axis=1)

    def set_w(self, w_type=3):
        if w_type == 0:
            self.w = [1.0*self.A[d] for d in range(self.D)]
        elif w_type == 1:
            self.w = [self.A[d]/(1.0*self.nlevel[d][:, np.newaxis]) for d in range(self.D)]
        elif w_type == 2:
            self.w = [self.A[d]/(1.0*self.n[d]) for d in range(self.D)]
        elif w_type == 3:
            self.w = [self.A[d]/(1.0*self.n[d]*self.nlevel[d][:, np.newaxis]) for d in range(self.D)]
        elif w_type == 4:
            node_degree = self.nlevel[0]
            if self.D > 1:
                node_degree = node_degree[:,np.newaxis] + self.nlevel[1]
                if self.D > 2:
                    node_degree = node_degree[:,:,np.newaxis] + self.nlevel[2]
                    if self.D > 3:
                        node_degree = node_degree[:,:,:,np.newaxis] + self.nlevel[3]
                        if self.D > 4:
                            print("Total number of features is larger than 4! Current implementation does not support it!")
            node_degree_inv = (1.0/self.N)/node_degree
            self.w = [None]*self.D
            for d in range(self.D):
                reduceD = tuple(range(d)+range(d+1, self.D))
                wtemp = np.sum(node_degree_inv, axis = reduceD)
                self.w[d] = self.A[d]*wtemp[:,np.newaxis]
        else:
            print("Unrecognizable weight type, please input again!")

    def prox_Fstar(self, Y, sigma):
        for d in range(self.D):
            # project every item to interval [-1,1]
            Y[d][self.A[d]] = map(lambda x: x/max(1, abs(x)), Y[d][self.A[d]])

    def apply_F(self, Y):
        tv = 0.
        for d in range(self.D):
            tv += np.sum(np.abs(Y[d][self.A[d]]))
        return tv

    def apply_K(self, x, Y):
        # x is the state variable, Y is the flux variable to store the result.
        # Y has the same dimension as A
        for d in range(self.D):
            xd = x[self.nind1[d]:self.nind1[d+1]]
            Y[d] = self.w[d]*(xd - xd[:, np.newaxis])

    def apply_Kstar(self, x, Y):
        x[0] = 0
        for d in range(self.D):
            # add the weights to every edge
            Ytemp = self.w[d]*Y[d]
            x[self.nind1[d]:self.nind1[d+1]] = np.sum(Ytemp, axis=0) - np.sum(Ytemp, axis=1)

    def get_L(self, n_it=10):
        '''
            Calculates the norm of operator K,
            i.e the sqrt of the largest eigenvalue of Kstar*K: ||K|| = sqrt(lambda_max(Kstar*K))
            data : acquired sinogram
        '''
        x = np.random.random(self.nt1)
        Y = [np.zeros((self.n[d], self.n[d])) for d in range(self.D)]

        for k in range(n_it):
            self.apply_K(x, Y)
            self.apply_Kstar(x, Y)
            s = np.linalg.norm(x)
            x /= s

        return np.sqrt(s)

    def PDALG2(self, Lambda, n_it, return_energy=True):
        '''
            # Lambda is the weight for the accuracy term
            Chambolle-Pock algorithm for the minimization of the objective function
                TV(x) + 1/2*Lambda*||A*x - d||_2^2
            Lambda : weight of the TV penalization (the higher Lambda, the more sparse is the solution)
            L : norm of the operator [P, Lambda*grad] (see power_method)
            n_it : number of iterations
            return_energy: if True, an array containing the values of the objective function will be returned
        '''

        L = self.get_L()
        sigma = 1/L
        tau = 1/L
        gamma = Lambda*self.Gamma[0]

        # primal dual splitting method
        # initialize
        # primal variable
        x = 1.0*self.lssol
        x_tilde = 1.0*x
        # dual variables
        Y = [np.zeros((self.n[d], self.n[d])) for d in range(self.D)]
        Y_tilde = [np.zeros((self.n[d], self.n[d])) for d in range(self.D)]

        if return_energy:
            en = np.zeros(n_it)
        for k in range(n_it):
            # update dual variables
            self.apply_K(x_tilde, Y_tilde)
            for d in range(self.D):
                Y[d] += sigma*Y_tilde[d]
            self.prox_Fstar(Y, sigma)
            # update primal variable
            x_old = 1.0*x
            self.apply_Kstar(x_tilde, Y)
            x -= tau*x_tilde
            # self.prox_G(x, tau)
            x += tau*Lambda*self.amg
            x = np.dot(self.U/(1+tau*Lambda*self.Gamma), np.dot(np.transpose(self.U), x))
            # update step size
            theta = 1.0
            # over-relaxation
            x_tilde = x + theta*(x - x_old)
            # print "theta, tau, sigma:", theta, tau, sigma
            # calculate norms
            if return_energy:
                # fidelity: no \|g\|^2 term
                fidelity = 0.5*np.dot(x, np.dot(self.ama0, x)) - np.dot(x, self.amg)
                # penalty
                x_res = [sum(x[self.nind1[d]:self.nind1[d + 1]]) for d in range(self.D)]
                penalty = 0.5*np.dot(self.mu, np.array(x_res)**2)
                # TV norm
                self.apply_K(x, Y_tilde)
                tv = self.apply_F(Y_tilde)
                energy = Lambda*(fidelity+penalty) + tv
                en[k] = energy
                if k % 10 == 0:
                    print("[%d] : energy %1.10e \t fidelity %e \t penalty %e \t TV %e" % (k, energy, fidelity, penalty, tv))

        # postprocessing
        u0 = x[0]
        u1 = [None]*self.D
        for d in range(self.D):
            u1[d] = x[self.nind1[d]:self.nind1[d + 1]]

        # return value
        if return_energy:
            return en, u0, u1
        else:
            return u0, u1
