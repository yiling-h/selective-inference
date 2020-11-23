import numpy as np
from pypet import Environment, cartesian_product
from selection.randomized.experiments.posterior_experiment import coverage_experiment


def main(nreps=1):
    # Create the environment
    env = Environment(trajectory='GrpLasso_Hetero',
                      comment='Randomized Group lasso, heterogeneous sizes and signals',
                      multiproc=True,
                      log_multiproc=True,
                      use_scoop=True,
                      wrap_mode='NETQUEUE',
                      overwrite_file=True,
                      filename='./hdf5/')

    # get the trajectory
    traj = env.traj

    # Now add the parameters with defaults
    traj.f_add_parameter('n', 500)
    traj.f_add_parameter('p', 100)
    traj.f_add_parameter('signal_fac', (np.float64(0.1), np.float64(1)))
    traj.f_add_parameter('groups', np.arange(10).repeat([2, 3, 4, 5, 7, 9, 10, 15, 20, 25]))
    traj.f_add_parameter('sgroup', [0, 1, 2])
    traj.f_add_parameter('sigma', 1)
    traj.f_add_parameter('rho', 0.35)
    traj.f_add_parameter('randomizer_scale', 0.71)
    traj.f_add_parameter('weight_frac', 1.0)
    traj.f_add_parameter('seed', 0)  # random seed

    seeds = [19860 + i for i in range(nreps)]  # offset seed for each rep

    # specify parameters to explore
    traj.f_explore(cartesian_product({"signal_fac": [(i, i+1) for i in np.linspace(0.5, 1.5, 10)],
                                      'seed': seeds}))

    env.run(coverage_experiment)

    env.disable_logging()


if __name__ == '__main__':
    # Let's make the python evangelists happy and encapsulate
    # the main function as you always should ;-)
    main(100)
