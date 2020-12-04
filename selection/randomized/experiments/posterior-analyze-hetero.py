from pypet import Trajectory
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

traj = Trajectory('GrpLasso_Hetero')

traj.f_load(filename='./selection/randomized/experiments/hdf5/GrpLasso_Hetero.hdf5',
            load_results=1, load_parameters=2)

coverage_naive = list(traj.f_get_from_runs(name='naive.mean.coverage', fast_access=True, auto_load=True, shortcuts=False).values())
length_naive = list(traj.f_get_from_runs(name='naive.mean.length', fast_access=True, auto_load=True, shortcuts=False).values())
nnz_naive = list(traj.f_get_from_runs(name='naive.nonzero.nnz', fast_access=True, auto_load=True, shortcuts=False).values())
tp_naive = list(traj.f_get_from_runs(name='naive.sigdet.tp', fast_access=True, auto_load=True, shortcuts=False).values())
tn_naive = list(traj.f_get_from_runs(name='naive.sigdet.tn', fast_access=True, auto_load=True, shortcuts=False).values())
fp_naive = list(traj.f_get_from_runs(name='naive.sigdet.fp', fast_access=True, auto_load=True, shortcuts=False).values())
fn_naive = list(traj.f_get_from_runs(name='naive.sigdet.fn', fast_access=True, auto_load=True, shortcuts=False).values())

coverage_split = list(traj.f_get_from_runs(name='split.mean.coverage', fast_access=True, auto_load=True, shortcuts=False).values())
length_split = list(traj.f_get_from_runs(name='split.mean.length', fast_access=True, auto_load=True, shortcuts=False).values())
nnz_split = list(traj.f_get_from_runs(name='split.nonzero.nnz', fast_access=True, auto_load=True, shortcuts=False).values())
tp_split = list(traj.f_get_from_runs(name='split.sigdet.tp', fast_access=True, auto_load=True, shortcuts=False).values())
tn_split = list(traj.f_get_from_runs(name='split.sigdet.tn', fast_access=True, auto_load=True, shortcuts=False).values())
fp_split = list(traj.f_get_from_runs(name='split.sigdet.fp', fast_access=True, auto_load=True, shortcuts=False).values())
fn_split = list(traj.f_get_from_runs(name='split.sigdet.fn', fast_access=True, auto_load=True, shortcuts=False).values())

coverage_posi = list(traj.f_get_from_runs(name='posi.mean.coverage', fast_access=True, auto_load=True, shortcuts=False).values())
length_posi = list(traj.f_get_from_runs(name='posi.mean.length', fast_access=True, auto_load=True, shortcuts=False).values())
nnz_posi = list(traj.f_get_from_runs(name='posi.nonzero.nnz', fast_access=True, auto_load=True, shortcuts=False).values())
tp_posi = list(traj.f_get_from_runs(name='posi.sigdet.tp', fast_access=True, auto_load=True, shortcuts=False).values())
tn_posi = list(traj.f_get_from_runs(name='posi.sigdet.tn', fast_access=True, auto_load=True, shortcuts=False).values())
fp_posi = list(traj.f_get_from_runs(name='posi.sigdet.fp', fast_access=True, auto_load=True, shortcuts=False).values())
fn_posi = list(traj.f_get_from_runs(name='posi.sigdet.fn', fast_access=True, auto_load=True, shortcuts=False).values())

signal_fac = traj.f_get('signal_fac').f_get_range()

df = pd.DataFrame({'Signal_Lower': [x[0] for x in signal_fac],
                   'coverage_naive': coverage_naive,
                   'coverage_split': coverage_split,
                   'coverage_posi': coverage_posi,
                   'length_naive': length_naive,
                   'length_split': length_split,
                   'length_posi': length_posi,
                   'nnz_naive': nnz_naive,
                   'nnz_split': nnz_split,
                   'nnz_posi' : nnz_posi,
                   'tp_naive' : tp_naive,
                   'tp_split' : tp_split,
                   'tp_posi' : tp_posi,
                   'tn_naive' : tn_naive,
                   'tn_split' : tn_split,
                   'tn_posi' : tn_posi,
                   'fp_naive' : fp_naive,
                   'fp_split' : fp_split,
                   'fp_posi' : fp_posi,
                   'fn_naive' : fn_naive,
                   'fn_split' : fn_split,
                   'fn_posi' : fn_posi,
                   })

df.to_csv('selection/randomized/experiments/hdf5/posterior-hetero.csv')
