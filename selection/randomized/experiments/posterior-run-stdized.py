import numpy as np
from pypet import Environment, cartesian_product
from selection.randomized.experiments.posterior_experiment import coverage_experiment


def main(nreps=1):
    # Create the environment
    env = Environment(trajectory='GrpLasso_Std_Balanced',
                      comment='Randomized Group lasso, Standardized, with balanced groups',
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
    traj.f_add_parameter('sigma', 3.)
    traj.f_add_parameter('rho', 0.20)
    traj.f_add_parameter('randomizer_scale', 0.71)
    traj.f_add_parameter('weight_frac', 1.0)
    traj.f_add_parameter('seed', 1986)  # random seed offset
    traj.f_add_parameter('rep', 0)  # dummy to track replications
    traj.f_add_parameter('std', True)  # standardized mode
    traj.f_add_parameter('og', False)  # overlapping groups mode
    traj.f_add_parameter('nsample', 1500)  # number of samples for posi

    # specify parameters to explore
    traj.f_explore(cartesian_product({"signal_fac": np.linspace(0.1, 1.5, 15),
                                      'sgroup': [3],
                                      'rep': range(nreps)}))

    env.run(coverage_experiment)

    env.disable_logging()


if __name__ == '__main__':
    # Let's make the python evangelists happy and encapsulate
    # the main function as you always should ;-)
    main(10)
