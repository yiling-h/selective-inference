import numpy as np
from selection.randomized.M_estimator import M_estimator

class M_estimator_exact(M_estimator):

    def __init__(self, loss, epsilon, penalty, randomization, randomizer='gaussian'):
        M_estimator.__init__(self, loss, epsilon, penalty, randomization)
        self.randomizer = randomizer

    def solve_approx(self):
        np.random.seed(0)
        self.solve()
        (_opt_linear_term, _opt_affine_term) = self.opt_transform
        self._opt_linear_term = np.concatenate(
            (_opt_linear_term[self._overall, :], _opt_linear_term[~self._overall, :]), 0)
        self._opt_affine_term = np.concatenate((_opt_affine_term[self._overall], _opt_affine_term[~self._overall]), 0)
        self.opt_transform = (self._opt_linear_term, self._opt_affine_term)

        (_score_linear_term, _) = self.score_transform
        self._score_linear_term = np.concatenate(
            (_score_linear_term[self._overall, :], _score_linear_term[~self._overall, :]), 0)
        self.score_transform = (self._score_linear_term, np.zeros(self._score_linear_term.shape[0]))
        self.feasible_point = np.abs(self.initial_soln[self._overall])
        lagrange = []
        for key, value in self.penalty.weights.iteritems():
            lagrange.append(value)
        lagrange = np.asarray(lagrange)
        self.inactive_lagrange = lagrange[~self._overall]

        X, _ = self.loss.data
        n, p = X.shape
        self.p = p

        nactive = self._overall.sum()
        score_cov = np.zeros((p, p))
        X_active_inv = np.linalg.inv(X[:,self._overall].T.dot(X[:,self._overall]))
        projection_perp = np.identity(n) - X[:,self._overall].dot(X_active_inv).dot( X[:,self._overall].T)
        score_cov[:nactive, :nactive] = X_active_inv
        score_cov[nactive:, nactive:] = X[:,~self._overall].T.dot(projection_perp).dot(X[:,~self._overall])

        self.score_target_cov = score_cov[:, :nactive]
        self.target_cov = score_cov[:nactive, :nactive]
        self.target_observed = self.observed_score_state[:nactive]
        self.nactive = nactive

        self.B_active = self._opt_linear_term[:nactive, :nactive]
        self.B_inactive = self._opt_linear_term[nactive:, :nactive]


    def setup_map(self, j):

        self.A = np.dot(self._score_linear_term, self.score_target_cov[:, j]) / self.target_cov[j, j]
        self.null_statistic = self._score_linear_term.dot(self.observed_score_state) - self.A * self.target_observed[j]

        self.offset_active = self._opt_affine_term[:self.nactive] + self.null_statistic[:self.nactive]
        self.offset_inactive = self.null_statistic[self.nactive:]

class M_estimator_2step(M_estimator):

    def __init__(self, loss, epsilon, penalty, randomization, simes_level, index, J, T_sign, threshold,
                 data_simes):
        M_estimator.__init__(self, loss, epsilon, penalty, randomization)
        self.simes_level = simes_level
        self.index = index
        self.J = J
        self.T_sign = T_sign
        self.threshold = threshold
        self.data_simes = data_simes
        self.nactive_simes = self.threshold.shape[0]

    def solve_approx(self):
        #map from lasso
        #np.random.seed(0)
        self.solve()
        (_opt_linear_term, _opt_affine_term) = self.opt_transform
        self._opt_linear_term = np.concatenate(
            (_opt_linear_term[self._overall, :], _opt_linear_term[~self._overall, :]), 0)
        self._opt_affine_term = np.concatenate((_opt_affine_term[self._overall], _opt_affine_term[~self._overall]), 0)
        self.opt_transform = (self._opt_linear_term, self._opt_affine_term)

        (_score_linear_term, _) = self.score_transform
        self._score_linear_term = np.concatenate(
            (_score_linear_term[self._overall, :], _score_linear_term[~self._overall, :]), 0)
        self.score_transform = (self._score_linear_term, np.zeros(self._score_linear_term.shape[0]))
        self.feasible_point_lasso = np.abs(self.initial_soln[self._overall])

        lagrange = []
        for key, value in self.penalty.weights.iteritems():
            lagrange.append(value)
        lagrange = np.asarray(lagrange)
        self.inactive_lagrange = lagrange[~self._overall]

        X, _ = self.loss.data
        n, p = X.shape
        self.p = p

        nactive = self._overall.sum()

        score_cov = np.zeros((p, p))
        X_active_inv = np.linalg.inv(X[:,self._overall].T.dot(X[:,self._overall]))
        projection_perp = np.identity(n) - X[:,self._overall].dot(X_active_inv).dot( X[:,self._overall].T)
        score_cov[:nactive, :nactive] = X_active_inv
        score_cov[nactive:, nactive:] = X[:,~self._overall].T.dot(projection_perp).dot(X[:,~self._overall])

        self.score_target_cov = score_cov[:, :nactive]
        self.target_cov = score_cov[:nactive, :nactive]
        self.target_observed = self.observed_score_state[:nactive]
        self.nactive = nactive

        self.B_active_lasso = self._opt_linear_term[:nactive, :nactive]
        self.B_inactive_lasso = self._opt_linear_term[nactive:, :nactive]


        if self.nactive_simes > 1:
            #print(self.nactive_simes, nactive)
            self.score_cov_simes = np.zeros((self.nactive_simes, nactive))
            self.score_cov_simes[0,:] = (X_active_inv.dot(X[:,self._overall].T).dot(X[:,self.index])).T
            self.score_cov_simes[1:,] = (X_active_inv.dot(X[:,self._overall].T).dot(X[:,self.J])).T
            #self.B_active_simes = np.zeros((1,1))
            #self.B_active_simes[0,0] = self.T_sign
            self.B_active_simes = np.identity(1) * self.T_sign
            self.B_inactive_simes = np.zeros((self.nactive_simes-1,1))
            self.inactive_threshold = self.threshold[1:]

        else:
            self.B_active_simes = np.identity(1) * self.T_sign
            self.score_cov_simes = (X_active_inv.dot(X[:, self._overall].T).dot(X[:, self.index]))
            self.inactive_threshold = -1

    def setup_map(self, j):

        self.A_lasso = np.dot(self._score_linear_term, self.score_target_cov[:, j]) / self.target_cov[j, j]
        self.null_statistic_lasso = self._score_linear_term.dot(self.observed_score_state) - self.A_lasso * self.target_observed[j]

        self.offset_active_lasso = self._opt_affine_term[:self.nactive] + self.null_statistic_lasso[:self.nactive]
        self.offset_inactive_lasso = self.null_statistic_lasso[self.nactive:]

        if self.nactive_simes > 1:
            linear_simes = np.zeros((self.nactive_simes, self.nactive_simes))
            linear_simes[0, 0] = -1.
            linear_simes[1:, 1:] = -np.identity(self.nactive_simes - 1)
            self.A_simes = np.dot(linear_simes, self.score_cov_simes[:, j]) / self.target_cov[j, j]
            self.null_statistic_simes = linear_simes.dot(self.data_simes) - self.A_simes * self.target_observed[j]

            self.offset_active_simes = self.T_sign * self.threshold[0] + self.null_statistic_simes[0]
            self.offset_inactive_simes = self.null_statistic_simes[1:]

        else:
            linear_simes = -1.
            #print("shapes", self.score_cov_simes[j, :].shape, self.target_cov[j, j].shape)
            self.A_simes = linear_simes* (self.score_cov_simes[j] / self.target_cov[j, j])
            self.null_statistic_simes = linear_simes* (self.data_simes) - self.A_simes * self.target_observed[j]
            self.offset_active_simes = self.T_sign * self.threshold[0] + self.null_statistic_simes


class M_estimator_2step_pruned(M_estimator):

    def __init__(self, loss, epsilon, penalty, randomization, simes_level, index, J, T_sign, threshold,
                 data_simes, X_simes):

        M_estimator.__init__(self, loss, epsilon, penalty, randomization)
        self.simes_level = simes_level
        self.index = index
        self.J = J
        self.T_sign = T_sign
        self.threshold = threshold
        self.data_simes = data_simes
        self.X_simes = X_simes
        self.nactive_simes = self.threshold.shape[0]

    def solve_approx(self):
        self.solve()
        (_opt_linear_term, _opt_affine_term) = self.opt_transform
        self._opt_linear_term = np.concatenate(
            (_opt_linear_term[self._overall, :], _opt_linear_term[~self._overall, :]), 0)
        self._opt_affine_term = np.concatenate((_opt_affine_term[self._overall], _opt_affine_term[~self._overall]), 0)
        self.opt_transform = (self._opt_linear_term, self._opt_affine_term)

        (_score_linear_term, _) = self.score_transform
        self._score_linear_term = np.concatenate(
            (_score_linear_term[self._overall, :], _score_linear_term[~self._overall, :]), 0)
        self.score_transform = (self._score_linear_term, np.zeros(self._score_linear_term.shape[0]))
        self.feasible_point_lasso = np.abs(self.initial_soln[self._overall])

        lagrange = []
        for key, value in self.penalty.weights.iteritems():
            lagrange.append(value)
        lagrange = np.asarray(lagrange)
        self.inactive_lagrange = lagrange[~self._overall]

        X, _ = self.loss.data
        n, p = X.shape
        self.p = p

        nactive = self._overall.sum()

        score_cov = np.zeros((p, p))
        X_active_inv = np.linalg.inv(X[:,self._overall].T.dot(X[:,self._overall]))
        projection_perp = np.identity(n) - X[:,self._overall].dot(X_active_inv).dot( X[:,self._overall].T)
        score_cov[:nactive, :nactive] = X_active_inv
        score_cov[nactive:, nactive:] = X[:,~self._overall].T.dot(projection_perp).dot(X[:,~self._overall])

        self.score_target_cov = score_cov[:, :nactive]
        self.target_cov = score_cov[:nactive, :nactive]
        self.target_observed = self.observed_score_state[:nactive]
        self.nactive = nactive

        self.B_active_lasso = self._opt_linear_term[:nactive, :nactive]
        self.B_inactive_lasso = self._opt_linear_term[nactive:, :nactive]


        if self.nactive_simes > 1:
            self.score_cov_simes = np.zeros((self.nactive_simes, nactive))
            self.score_cov_simes[0,:] = (X_active_inv.dot(X[:,self._overall].T).dot(self.X_simes[:,0])).T
            self.score_cov_simes[1:,] = (X_active_inv.dot(X[:,self._overall].T).dot(self.X_simes[:,1:])).T
            self.B_active_simes = np.identity(1) * self.T_sign
            self.B_inactive_simes = np.zeros((self.nactive_simes-1,1))
            self.inactive_threshold = self.threshold[1:]

        else:
            self.B_active_simes = np.identity(1) * self.T_sign
            self.score_cov_simes = (X_active_inv.dot(X[:, self._overall].T).dot(self.X_simes[:, 0]))
            self.inactive_threshold = -1

    def setup_map(self, j):

        self.A_lasso = np.dot(self._score_linear_term, self.score_target_cov[:, j]) / self.target_cov[j, j]
        self.null_statistic_lasso = self._score_linear_term.dot(self.observed_score_state) - self.A_lasso * self.target_observed[j]

        self.offset_active_lasso = self._opt_affine_term[:self.nactive] + self.null_statistic_lasso[:self.nactive]
        self.offset_inactive_lasso = self.null_statistic_lasso[self.nactive:]

        if self.nactive_simes > 1:
            linear_simes = np.zeros((self.nactive_simes, self.nactive_simes))
            linear_simes[0, 0] = -1.
            linear_simes[1:, 1:] = -np.identity(self.nactive_simes - 1)
            self.A_simes = np.dot(linear_simes, self.score_cov_simes[:, j]) / self.target_cov[j, j]
            self.null_statistic_simes = linear_simes.dot(self.data_simes) - self.A_simes * self.target_observed[j]

            self.offset_active_simes = self.T_sign * self.threshold[0] + self.null_statistic_simes[0]
            self.offset_inactive_simes = self.null_statistic_simes[1:]

        else:
            linear_simes = -1.
            self.A_simes = linear_simes* (self.score_cov_simes[j] / self.target_cov[j, j])
            self.null_statistic_simes = linear_simes* (self.data_simes) - self.A_simes * self.target_observed[j]
            self.offset_active_simes = self.T_sign * self.threshold[0] + self.null_statistic_simes

