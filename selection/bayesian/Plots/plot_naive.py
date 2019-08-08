import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def common_format(ax):
    ax.grid(True, which='both')
    ax.set_xlabel('signal regimes', fontsize=16)
    ax.set_ylabel('', fontsize=16)
    return ax

def plot_selective_metrics(infile1, infile2, infile3, outpath):

    df = pd.read_csv(infile1)
    df = df[df['nactive'] > 0.]

    df_1 = pd.read_csv(infile2)
    #df_1 = df_1.append(pd.read_csv(infile3))
    df_1 = df_1[df_1['snr'] < 3.]

    df_2 = df
    df_2 = df_2.append(df_1)

    df_1 = df_1.groupby(['snr', 'simulation'], as_index=False).mean()

    snr_alias = {0.1: 1,
                 0.50: 2,
                 1.0: 3,
                 2.: 4}

    df['snr'] = df['snr'].map(snr_alias)
    df_1['snr'] = df_1['snr'].map(snr_alias)
    df_2['snr'] = df_2['snr'].map(snr_alias)

    cols = ["tomato"]
    cols_1 = ["deepskyblue"]
    sns.set(font_scale=1.6) # fond size
    sns.set_style("white", {'axes.facecolor': 'white',
                            'axes.grid': True,
                            'axes.linewidth': 2.0,
                            'grid.linestyle': u'--',
                            'grid.linewidth': 4.0,
                            'xtick.major.size': 5.0,
                          })

    fig = plt.figure(figsize=(12, 4))
    ax3 = fig.add_subplot(133)
    ax2 = fig.add_subplot(132)
    ax1 = fig.add_subplot(131)

    order = ["selective", "naive"]
    cols_3 = ["deepskyblue", "tomato"]

    sns.boxplot(x="snr", y="coverage", data=df, ax=ax1, palette=cols)
    sns.stripplot(x="snr", y="coverage", data=df, ax=ax1, jitter = True, size=4,color="maroon")

    sns.barplot(x="snr", y="coverage", data=df_2, hue_order=order, hue="method", ax=ax2, palette=cols_3)

    sns.boxplot(x="snr", y="coverage", data=df_1, ax=ax3, palette=cols_1)
    sns.stripplot(x="snr", y="coverage", data=df_1, ax=ax3, jitter=True, size=4, color="navy")

    ax1.set_ylim(-0.01, 1.02)
    ax2.set_ylim(-0.01, 1.02)
    ax3.set_ylim(-0.01, 1.02)

    ax2.legend_.remove()

    common_format(ax1)
    common_format(ax2)
    common_format(ax3)
    ax1.set_title('', y=1.01, size=16)
    ax2.set_title('', y=1.01, size=16)
    ax2.axhline(y=0.90, color='k', linestyle='--', linewidth=2)
    ax3.set_title('', y=1.01, size=16)

    fig.suptitle('coverage: naive vs selective', y=1.05, x=0.52, size=18)

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    outfile = os.path.join(outpath, "naive_comparison_35_real_90.pdf")
    plt.savefig(outfile, format='pdf', bbox_inches='tight')
