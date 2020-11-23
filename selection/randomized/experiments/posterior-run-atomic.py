import numpy as np
from pypet import Environment, cartesian_product
from .posterior_experiment import coverage_experiment


def main(nreps=1):
    # Create the environment
    env = Environment(trajectory='GrpLasso_Singletons',
                      comment='Randomized Group lasso, but each group is atomic',
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
    traj.f_add_parameter('signal_fac', 0.1)
    traj.f_add_parameter('groups', np.arange(100).repeat(1))
    traj.f_add_parameter('sgroup', 5)
    traj.f_add_parameter('sigma', 1)
    traj.f_add_parameter('rho', 0.35)
    traj.f_add_parameter('randomizer_scale', 0.71)
    traj.f_add_parameter('weight_frac', 1.0)
    traj.f_add_parameter('seed', 0)  # random seed

    seeds = [19860 + i for i in range(nreps)]  # offset seed for each rep

    # specify parameters to explore
    traj.f_explore(cartesian_product({"signal_fac": np.linspace(0.5, 1.5, 10),
                                      'sgroup': [5],
                                      'seed': seeds}))

    env.run(coverage_experiment)

    env.disable_logging()


if __name__ == '__main__':
    # Let's make the python evangelists happy and encapsulate
    # the main function as you always should ;-)
    main(100)
