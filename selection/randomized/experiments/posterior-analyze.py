from pypet import Trajectory
import matplotlib.pyplot as plt
import matplotlib

traj = Trajectory('CoverageChecks')

traj.f_load(filename='./selection/randomized/experiments/hdf5/CoverageChecks.hdf5',
            load_results=1, load_parameters=2)

traj.f_get_from_runs(name='mean.coverage', fast_access=True, auto_load = True, shortcuts = False)

traj.f_get_from_runs(name='mean.length', fast_access=True, auto_load = True, shortcuts = False)
