from pypet import Trajectory
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

traj = Trajectory('CoverageChecks')

traj.f_load(filename='./selection/randomized/experiments/hdf5/CoverageChecks.hdf5',
            load_results=2, load_parameters=2)

coverage_posi = list(traj.f_get_from_runs(name='posi.mean.coverage', fast_access=True, auto_load=True, shortcuts=False).values())
length_posi = list(traj.f_get_from_runs(name='posi.mean.length', fast_access=True, auto_load=True, shortcuts=False).values())
coverage_naive = list(traj.f_get_from_runs(name='naive.mean.coverage', fast_access=True, auto_load=True, shortcuts=False).values())
length_naive = list(traj.f_get_from_runs(name='naive.mean.length', fast_access=True, auto_load=True, shortcuts=False).values())
coverage_split = list(traj.f_get_from_runs(name='split.mean.coverage', fast_access=True, auto_load=True, shortcuts=False).values())
length_split = list(traj.f_get_from_runs(name='split.mean.length', fast_access=True, auto_load=True, shortcuts=False).values())

# ugly hack b/c run5 for naive inference selected nothing; duplicate the run that worked
coverage_naive = coverage_naive[0:5] + coverage_naive[4:]
length_naive = length_naive[0:5] + length_naive[4:]

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


dfg = df.groupby(['Signal', 'SGroup'])

dfg.agg([np.size, np.mean, np.std])

dfg.mean()
dfg.var()
dfg.size()


df.groupby(['Signal', 'SGroup']).var()

df.groupby(['Signal', 'SGroup'])

for sg in np.unique(sgroup):
    mask = np.equal(sg, sgroup)
    plt.plot(np.array(signal_fac)[mask], np.array(list(coverage.values()))[mask], label = 'Number Active Groups: %s' % str(sg))

plt.xlabel('Signal')
plt.ylabel('Coverage')
plt.legend()

plt.savefig('plot.png')
