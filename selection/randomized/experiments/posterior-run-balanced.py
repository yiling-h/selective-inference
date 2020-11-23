import numpy as np
from pypet import Environment, cartesian_product
from .posterior_experiment import coverage_experiment


def main(nreps=1):
    # Create the environment
    env = Environment(trajectory='GrpLasso_Balanced',
                      comment='Randomized Group lasso, with balanced groups',
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
    traj.f_add_parameter('signal_fac', np.float64(0.))
    traj.f_add_parameter('groups', np.arange(25).repeat(4))
    traj.f_add_parameter('sgroup', 3)
    traj.f_add_parameter('sigma', 1)
    traj.f_add_parameter('rho', 0.35)
    traj.f_add_parameter('randomizer_scale', 0.3)
    traj.f_add_parameter('weight_frac', 1.0)
    traj.f_add_parameter('seed', 0)  # random seed

    seeds = [1986 + i for i in range(nreps)]  # offset seed for each rep

    # specify parameters to explore
    traj.f_explore(cartesian_product({"signal_fac": np.arange(0.1, 2, 0.1),
                                      'sgroup': [3],
                                      'seed': seeds}))

    env.run(coverage_experiment)

    env.disable_logging()


if __name__ == '__main__':
    # Let's make the python evangelists happy and encapsulate
    # the main function as you always should ;-)
    main(100)
