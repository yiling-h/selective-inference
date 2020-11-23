from pypet import Trajectory
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

traj = Trajectory('GrpLasso_Balanced')

traj.f_load(filename='./selection/randomized/experiments/hdf5/GrpLasso_Balanced.hdf5',
            load_results=1, load_parameters=2)

coverage_posi = list(traj.f_get_from_runs(name='posi.mean.coverage', fast_access=True, auto_load=True, shortcuts=False).values())
length_posi = list(traj.f_get_from_runs(name='posi.mean.length', fast_access=True, auto_load=True, shortcuts=False).values())
coverage_naive = list(traj.f_get_from_runs(name='naive.mean.coverage', fast_access=True, auto_load=True, shortcuts=False).values())
length_naive = list(traj.f_get_from_runs(name='naive.mean.length', fast_access=True, auto_load=True, shortcuts=False).values())
coverage_split = list(traj.f_get_from_runs(name='split.mean.coverage', fast_access=True, auto_load=True, shortcuts=False).values())
length_split = list(traj.f_get_from_runs(name='split.mean.length', fast_access=True, auto_load=True, shortcuts=False).values())

sgroup = traj.f_get('sgroup').f_get_range()
signal_fac = traj.f_get('signal_fac').f_get_range()

df = pd.DataFrame({'SGroup': sgroup,
                   'Signal': signal_fac,
                   'coverage_naive': coverage_naive,
                   'coverage_split': coverage_split,
                   'coverage_posi': coverage_posi,
                   'length_naive': length_naive,
                   'length_split': length_split,
                   'length_posi': length_posi})

df.to_csv('selection/randomized/experiments/posterior-balanced.csv')

