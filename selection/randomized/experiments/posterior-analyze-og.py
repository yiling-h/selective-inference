from pypet import Trajectory
from selection.randomized.experiments.posterior_analyze import make_df

traj = Trajectory('GrpLasso_OG_Balanced')

traj.f_load(filename='./selection/randomized/experiments/hdf5/GrpLasso_OG_Balanced.hdf5',
            load_results=1, load_parameters=2)

df = make_df(traj)

df.to_csv('selection/randomized/experiments/hdf5/posterior-og.csv')
