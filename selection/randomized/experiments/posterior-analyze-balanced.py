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

df.to_csv('posterior-balanced.csv')

# below here are hacky attempts to make plots

import seaborn as sns
sns.set_theme(style="darkgrid")


dfg = df.groupby(['Signal', 'SGroup'])

dfga = dfg.agg([np.size, np.mean, np.std])

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

# make a figure: coverage by signal for sgroup = 3, trace by method
dfm = df[df['SGroup'] == 3]

plt.figure()
plt.plot(dfm['Signal'],dfm['coverage_naive'],label='Naive')
plt.plot(dfm['Signal'],dfm['coverage_split'],label='Split')
plt.plot(dfm['Signal'],dfm['coverage_posi'],label='PoSI')

plt.legend(title='Method')
plt.xlabel('Signal')
plt.ylabel('Coverage')
plt.title('Coverage by Method, 3 Active Groups')
# plt.show()
plt.savefig('/home/kesslerd/repos/og-posi-manuscript/figs/coverage-sgroup3.png')

plt.figure()
plt.plot(dfm['Signal'],dfm['length_naive'],label='Naive')
plt.plot(dfm['Signal'],dfm['length_split'],label='Split')
plt.plot(dfm['Signal'],dfm['length_posi'],label='PoSI')

plt.legend(title = 'Method')
plt.xlabel('Signal')
plt.ylabel('Length')
plt.title('CI Length by Method, 3 Active Groups')
# plt.show()
plt.savefig('/home/kesslerd/repos/og-posi-manuscript/figs/length-sgroup3.png')


