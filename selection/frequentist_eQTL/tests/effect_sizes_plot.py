import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pfxes = ["proposed", "fwd_bwd"]
level = 0.10
df = pd.DataFrame()
# load all data
for pfx in pfxes:
    fn = os.path.join('/Users/snigdhapanigrahi/Dropbox/Selective inference EQTLs/results/real_data/', "effect_sizes_{}_{:.2f}.csv".format(pfx, level))
    print(fn)
    df_sub = pd.read_csv(fn)
    df = df.append(df_sub)

print("Result dimension: {}".format(df.shape))
print("Methods: {}".format(np.unique(df['method'])))
alias = {"sel_inf": "Adjusted", "fwd_bwd": "Naive"}
df['method'] = df['method'].map(alias)
df = df.loc[df["MAF"] < 0.1]
df_Adjusted = df.loc[df["method"] == "Adjusted"]
df_Aguet = df.loc[df["method"] == "Naive"]
df.head()

#cols = ["#3498db", "#e74c3c"]
cols = ["#008b8b", "#b0171f"]
sns.set(font_scale=1.8)  # fond size
sns.set_style("white", {'axes.facecolor': 'white',
                        'axes.grid': True,
                        'axes.linewidth': 2.0,
                        'grid.linestyle': u'--',
                        'grid.linewidth': 4.0,
                        'xtick.major.size': 5.0,
                        })

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.scatter(df_Adjusted['MAF'], df_Adjusted['effect_size'], s=50,
            color=cols[0], edgecolors=cols[0], alpha=0.7)
ax2.scatter(df_Aguet['MAF'], df_Aguet['effect_size'], s=50,
            color=cols[1], edgecolors=cols[1], alpha=0.7)


def common_format(ax):
    fontsize = 22
    ax.set_xlabel('minor allele frequency', fontsize=fontsize)
    ax.set_ylabel('effect size', fontsize=fontsize)
    ax.set_xlim(0.05, 0.10)
    ax.set_ylim(-15, 15)
    return ax


def set_title(method):
    num_vars = np.sum(df['method'] == method)
    return "{} ({} eVariants)".format(method, num_vars)


common_format(ax1)
common_format(ax2)

ax1.set_title(set_title('Adjusted'))
ax2.set_title(set_title('Naive'))

plt.tight_layout(pad=1.4, w_pad=0.5, h_pad=1.0)
plt.savefig(os.path.join('/Users/snigdhapanigrahi/Documents/Research/Job Talk/',
                         'effect_my_size_comparison_{:.2f}.pdf'.format(level)), format='pdf', bbox_inches='tight')