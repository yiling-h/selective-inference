import numpy as np
from selection.randomized.tests.test_group_lasso_posterior import test_posterior_inference
from pypet import Environment, cartesian_product
from selection.tests.instance import gaussian_group_instance
from selection.randomized.group_lasso import group_lasso, posterior


def posterior_coverage(traj):
    np.random.seed(seed=traj.seed)

    inst = gaussian_group_instance

    signal = np.sqrt(traj.signal_fac * 2 * np.log(traj.p))

    X, Y, beta = inst(n=traj.n,
                      p=traj.p,
                      signal=signal,
                      groups=traj.groups,
                      sgroup=traj.sgroup,
                      sigma=traj.sigma,
                      rho=traj.rho,
                      equicorrelated=False,
                      random_signs=True)[:3]

    traj.f_add_result('data.X', X)
    traj.f_add_result('data.Y', Y)
    traj.f_add_result('data.beta', beta)

    sigma_ = np.std(Y)          # sigma-hat
    if traj.n > traj.p:
        dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (traj.n - traj.p)
    else:
        dispersion = sigma_ ** 2

    sigma_ = np.sqrt(dispersion)

    weights = dict([(i, traj.weight_frac * sigma_ * np.sqrt(2 * np.log(traj.p))) for i in np.unique(traj.groups)])
    conv = group_lasso.gaussian(X,
                                Y,
                                traj.groups,
                                weights,
                                randomizer_scale=traj.randomizer_scale * sigma_)

    signs, _ = conv.fit()
    nonzero = signs != 0
    conv._setup_implied_gaussian()

    def prior(target_parameter, prior_var=100):
        grad_prior = -target_parameter / prior_var
        log_prior = -np.linalg.norm(target_parameter) ** 2 / (2. * prior_var)
        return grad_prior, log_prior

    if nonzero.sum() > 0:       # this is too dense
        beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

        posterior_inf = posterior(conv,  #  this sometimes takes a long time to run
                                  prior=prior,
                                  dispersion=dispersion)

        samples = posterior_inf.langevin_sampler(nsample=1500,
                                                 nburnin=100,
                                                 step=1.,
                                                 verbose=0)

        traj.f_add_result('samples', samples)

        lci = np.percentile(samples, 5, axis=0)
        uci = np.percentile(samples, 95, axis=0)
        coverage = (lci < beta_target) * (uci > beta_target)
        length = uci - lci

        traj.f_add_result('componentwise.coverage', coverage)
        traj.f_add_result('componentwise.length', length)

        traj.f_add_result('mean.coverage', np.mean(coverage))
        traj.f_add_result('mean.length', np.mean(length))


def main():
    # Create the environment
    env = Environment(trajectory = 'CoverageChecks',
                      comment='Our first pypet experiment',
                      multiproc=True,
                      ncores=10,
                      overwrite_file=True,
                      filename='./hdf5/')

    # get the trajectory
    traj = env.traj

    # Now add the parameters with defaults
    traj.f_add_parameter('n', 500)
    traj.f_add_parameter('p', 200)
    traj.f_add_parameter('signal_fac', 0.1)
    traj.f_add_parameter('groups', np.arange(50).repeat(4))
    traj.f_add_parameter('sgroup', 3)
    traj.f_add_parameter('sigma', 3)
    traj.f_add_parameter('rho', 0.3)
    traj.f_add_parameter('randomizer_scale', 0.3)
    traj.f_add_parameter('weight_frac', 1.0)
    traj.f_add_parameter('seed', 0)  # random seed

    # specify parameters to explore
    traj.f_explore(cartesian_product({"signal_fac": [0.1, 0.2, 0.3],
                                      'sgroup': [3, 4],
                                      'seed': [1986, 2020, 2001]}))

    env.run(posterior_coverage)

    env.disable_logging()


if __name__ == '__main__':
    # Let's make the python evangelists happy and encapsulate
    # the main function as you always should ;-)
    main()
