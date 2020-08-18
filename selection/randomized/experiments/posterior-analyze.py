from pypet import Trajectory
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

traj = Trajectory('CoverageChecks')

traj.f_load(filename='./hdf5/CoverageChecks.hdf5',
            load_results=2, load_parameters=2)

coverage = traj.f_get_from_runs(name='mean.coverage', fast_access=True, auto_load=True, shortcuts=False)

length = traj.f_get_from_runs(name='mean.length', fast_access=True, auto_load = True, shortcuts = False)

sgroup = traj.f_get('sgroup').f_get_range()
signal_fac = traj.f_get('signal_fac').f_get_range()

df = pd.DataFrame({'SGroup': sgroup,
                   'Signal': signal_fac,
                   'coverage': list(coverage.values()),
                   'length': list(length.values())})

df.groupby(['SGroup', 'Signal']).mean()
df.groupby(['SGroup', 'Signal']).var()

for sg in np.unique(sgroup):
    mask = np.equal(sg, sgroup)
    plt.plot(np.array(signal_fac)[mask], np.array(list(coverage.values()))[mask], label = 'Number Active Groups: %s' % str(sg))

plt.xlabel('Signal')
plt.ylabel('Coverage')
plt.legend()

plt.savefig('plot.png')
