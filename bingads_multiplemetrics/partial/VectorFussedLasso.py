import numpy as np
import scipy.optimize as sciopt
import matplotlib.pyplot as plt


class VectorFussedLasso(object):
    def __init__(self, g, m, dates):
        # g and m are 2-dimensional arrays with the same shape: T * P
        # g is the point-wise averaged noisy measurements (for every metric at every time point)
        # m is the effective percentage of measurements at the corresponding point (inverse of variance)
        #   min_u \|D u\| + Lambda/2 \|u - g\|_{f,m}^2
        self.g = g
        self.T, self.P = np.array(g.shape)
        self.N = self.T * self.P
        self.m = m/self.N           #normalization, it is not necessary to do this.
        self.mg = self.m * self.g
        self.dates = dates

        # constant effect
        u0 = np.sum(self.mg, axis=0)/np.sum(self.m, axis=0)
        self.U0 = np.tile(u0, (self.T, 1))
        self.c0 = np.sum(self.m * (self.g - self.U0)**2)
        # fidelity terms
        # self.B = diag(self.m) now, since we A = I now
        self.b = self.m * self.g
        self.c = np.sum(self.g * self.b)

        # weights for different metrics when consider its distance: default all equal weights
        self.w_metric = np.ones(self.P)
        self.w_constant = True

    def set_w_metric(self, w):
        self.w_metric = w
        self.w_constant = False

    def apply_F(self, Y):
        tv = 0.
        for y in Y:
            tv += np.linalg.norm(y)
        return tv

    def prox_Fstar(self, Y, Sigma):
        # argmin_x (x-y)^T Simga^{-1} (x-y)/2 + \delta_{max_t \|x_t\|_2 \le 1}
        if self.w_constant:
            for i, y in enumerate(Y):
                # project every item to ball \|x\|_2 \le 1
                ynorm = np.linalg.norm(y)
                if ynorm > 1:
                    Y[i] = y/ynorm
                # else, do nothing
        else:
            for i, y in enumerate(Y):
                # project every item to ball \|x\|_2 \le 1
                if np.linalg.norm(y) > 1:
                    lamb = sciopt.newton(myfunc, 0.0, myfuncprime, args=(y, Sigma[i]))
                    Y[i] = y / (1 + lamb*Sigma[i])
                # else, do nothing

    def get_flux(self, U):
        flux = np.zeros(self.T-1)
        for i in range(self.T-1):
            flux[i] = np.linalg.norm((U[i+1, ] - U[i, ]) * self.w_metric)
        return flux

    def apply_Kstar(self, U, Y):
        U[:-1, ] = - Y
        U[-1, ] = 0
        U[1:, ] += Y
        U *= self.w_metric

    def get_Tinv_Sigma(self):
        tinv = np.ones(self.T)
        tinv[1:-1] = 2.0
        Tinv = tinv[:, np.newaxis] * self.w_metric

        sigma = np.ones(self.T-1)
        Sigma = 0.5 * sigma[:, np.newaxis] * (1./self.w_metric)

        return Tinv, Sigma

    def P_PDALG1(self, Lambda, n_it, return_energy=True):
        '''
            # Lambda is the weight for the accuracy term
            Preconditioned Chambolle-Pock algorithm for the minimization of the objective function
                \|D x\|_1 + 1/2*Lambda*||B*x - b||_2^2
            Lambda : weight of the TV penalization (the higher Lambda, the more sparse is the solution)
            n_it : number of iterations
            return_energy: if True, an array containing the values of the objective function will be returned
        '''
        theta = 1.0
        Tinv, Sigma = self.get_Tinv_Sigma()

        # primal dual splitting method
        # initialize
        # primal variable
        X = 1.0*self.U0
        X_tilde = 1.0*X
        # dual variables
        Y = np.zeros((self.T-1, self.P))
        Y_tilde = 1.0*Y

        if return_energy:
            en = np.zeros(n_it)
        for k in range(n_it):
            # calculate norms
            if return_energy:
                # fidelity
                fidelity = 0.5 * np.sum(self.m * (self.g - X)**2)
                # TV norm
                # self.apply_K(X, Y_tilde)
                Y_tilde = (X[1:, ] - X[:-1, ]) * self.w_metric
                tv = self.apply_F(Y_tilde)
                # energy
                energy = Lambda*fidelity + tv
                en[k] = energy
                if k % 10 == 0:
                    print("[%d] : energy %1.10e \t fidelity %e \t TV %e" % (k, energy, fidelity, tv))

            # update dual variables
            # self.apply_K(X_tilde, Y_tilde)
            Y_tilde = (X_tilde[1:, ] - X_tilde[:-1, ]) * self.w_metric
            Y += Sigma * Y_tilde
            self.prox_Fstar(Y, Sigma)
            # update primal variable
            X_old = 1.0 * X
            self.apply_Kstar(X_tilde, Y)
            X = Lambda * self.b + Tinv * X_old - X_tilde
            # self.prox_G(x, tau)
            X /= Tinv + Lambda * self.m
            X_tilde = X + theta * (X - X_old)

        # return value
        if return_energy:
            return en, X
        else:
            return X

    def P_PDALG2(self, lam, b, Tinv, Sigma, theta, X, Y, tol=1e-8, max_iter=30000):
        # creat auxillary variables
        X_tilde = 1.0*X
        # dual variables
        Y_tilde = 1.0*Y
        # energy monitoring
        en = np.zeros(max_iter//100)
        # run pre-conditioned primal dual algorithm
        for k in range(max_iter):
            # update dual variables
            Y_tilde = (X_tilde[1:, ] - X_tilde[:-1, ]) * self.w_metric
            Y += Sigma * Y_tilde
            self.prox_Fstar(Y, Sigma)
            # update primal variable
            X_old = 1.0 * X
            self.apply_Kstar(X_tilde, Y)
            X = lam * b + Tinv * X_old - X_tilde
            # self.prox_G(x, tau)
            X /= Tinv + lam * self.m
            X_tilde = X + theta * (X - X_old)
            if k % 100 == 0:
                # energyid
                energyid = k/100
                # fidelity: no \|g\|^2 term
                fidelity = 0.5 * np.sum(self.m * (self.g - X)**2)
                # TV norm
                Y_tilde = (X[1:, ] - X[:-1, ]) * self.w_metric
                tv = self.apply_F(Y_tilde)
                # energy
                energy = lam * fidelity + tv
                en[energyid] = energy
                # check the stopping criteria
                if energyid > 5 and abs(en[energyid-1]-en[energyid]) < tol*(en[0]-en[1]):
                    break
        return k, X, Y

    def lasso_path(self, lam_min, lam_max, nstep=100, tol=1e-8):
        # the list of lambdas we will compute
        lam_list = 1.0/np.linspace(1.0/lam_min, 1.0/lam_max, nstep)
        # settings of the P_PDALG
        theta = 1.0
        Tinv, Sigma = self.get_Tinv_Sigma()
        # initial guess for the very beginning
        # primal variable
        X = 1.0*self.U0
        # dual variables
        Y = np.zeros((self.T - 1, self.P))

        # compute lasso path
        u_list = [None]*nstep
        u_list[0] = 1.0*X
        noise_energy_list = [0.5*self.c0]
        uls_list = [1.0*X]
        for i, lam in enumerate(lam_list):
            # store last step solution
            X_old = 1.0 * X
            flux_old = self.get_flux(X_old)
            num_iter, X, Y = self.P_PDALG2(lam, self.b, Tinv, Sigma, theta, X_old, Y, tol=1e-8, max_iter=10000)
            u_list[i] = 1.0 * X
            # if change, do something
            flux = self.get_flux(X)
            is_changed = ((np.abs(flux) > 2*tol*np.sqrt(self.P)) == (np.abs(flux_old) > 2*tol*np.sqrt(self.P)))
            if not(is_changed.all()):
                # support changes
                X_ls, num_groups, noise_energy = self.blockLS(X, flux, tol=tol)
                print "Case: lambdaId, Lamda = ", i, lam
                print "Number of non-zero groups = ", num_groups
                print "num_iter, Noise_Energy", num_iter, noise_energy
                self.print_effect(X_ls, lam, i, "LassoPath", xlabel=self.dates)
                uls_list.append(1.0*X_ls)
                noise_energy_list.append(noise_energy)

        return u_list, noise_energy_list, uls_list

    def blockLS(self, X, flux, tol):
        clots = np.nonzero(np.abs(flux) > 2*tol*np.sqrt(self.P))[0]
        ngroups = len(clots) + 1
        utrans = np.zeros((self.T, ngroups))
        utrans[:clots[0] + 1, 0] = 1
        for groupid in range(1, ngroups-1):
            utrans[clots[groupid-1] + 1:clots[groupid] + 1, groupid] = 1
        utrans[clots[ngroups-2]+1:, ngroups-1] = 1
        if not (np.sum(utrans) == self.T):
            print "There's something wrong in clustering!"
        # form the piecewise constant solution
        a = np.zeros((ngroups, self.P))
        for groupid in range(ngroups):
            grouppts = (utrans[:, groupid] > 0.5)
            a[groupid] = np.sum(self.m[grouppts, :]*self.g[grouppts, :], axis=0) / np.sum(self.m[grouppts, :], axis=0)
        uls = np.dot(utrans, a)
        noise_energy = 0.5 * np.sum(self.m * (self.g - uls)**2)
        return uls, ngroups, noise_energy

    def print_effect(self, u, lam, stepid, str_method, xlabel):
        # plot computed result
        fig, ax = plt.subplots()
        plt.plot(range(self.T), u)
        plt.xlim([-1, self.T])
        ax.set_xticks(np.arange(self.T), minor=False)
        ax.set_xticklabels(xlabel, minor=False, rotation=45)
        plt.title("TV-regression: Lambda=%1.2e" % lam)
        fig.autofmt_xdate()
        # plt.show()
        plt.savefig('figure/'+str_method+'/Step%d_Lambda%1.2e.png' % (stepid, lam))
        plt.close()


def myfunc(x, y, sigma):
    # 1 - \sum_i y_i^2/(1 + x sigma_i)^2 = 0; ; we know np.linalg.norm(y) > 1
    return 1 - np.sum(y**2 / (1 + x*sigma)**2)

def myfuncprime(x, y, sigma):
    return 2 * np.sum(sigma * y**2 / (1 + x*sigma)**3)
