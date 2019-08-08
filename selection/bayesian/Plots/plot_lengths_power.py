import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def common_format(ax):
    ax.grid(True, which='both')
    ax.set_xlabel('signal regimes', fontsize=14)
    ax.set_ylabel('', fontsize=14)
    return ax

def common_format_1(ax):
    ax.grid(True, which='both')
    ax.set_xlabel('number of signals', fontsize=14)
    ax.set_ylabel('', fontsize=14)
    return ax

def compute_power(row):
    if row['true_total']>= 1. and row['true_total']<=3.:
        return 1
    elif row['true_total']>3. and row['true_total']<=6:
        return 2
    else:
        return 3

def plot_selective_metrics(infile1, infile2, infile3, outpath):

    df = pd.read_csv(infile2)
    df = df.append(pd.read_csv(infile3))

    snr_alias = {0.1: 1,
                 0.5: 2,
                 1.0: 3,
                 2.: 4}

    df = df[df['snr'] < 3.]
    df['snr'] = df['snr'].map(snr_alias)
    df_0 = df[df['nactive'] > 0.]
    df = df[df['true_total'] > 0.]
    df['regime'] = df.apply(compute_power, axis=1)

    df = df.groupby(['method', 'snr', 'simulation'], as_index=False).mean()

    cols = ["deepskyblue", "peachpuff", "lightpink", "plum", "palevioletred", "mediumpurple"]
    order = ["selective", "split (50%)", "split (60%)", "split (70%)", "split (80%)", "split (90%)"]
    sns.set(font_scale=1.)  # fond size
    sns.set_style("white", {'axes.facecolor': 'white',
                            'axes.grid': True,
                            'axes.linewidth': 2.0,
                            'grid.linestyle': u'--',
                            'grid.linewidth': 4.0,
                            'xtick.major.size': 4.0,
                            'ytick.major.size': 4.0
                            })

    fig = plt.figure(figsize=(12, 3.8))
    # ax3 = fig.add_subplot(133)
    ax2 = fig.add_subplot(132)
    ax1 = fig.add_subplot(131)


    sns.barplot(x="snr", y="length", hue_order=order, hue="method", data=df_0, ax=ax2, palette=cols)
    sns.barplot(x="regime", y="power_screen", hue_order=order, hue="method", data=df, ax=ax1, palette=cols)
    #sns.swarmplot(x="snr", y="power_screen", hue_order=order, hue="method", data=df, ax=ax2, palette=cols)
    #sns.stripplot(x="snr", y="power_screen", data=df, ax=ax2, jitter=True, size=4, palette=cols)

    ax1.set(xticklabels=['1-3', '4-6', '>6'])
    ax1.legend_.remove()
    ax2.legend(loc='center left',bbox_to_anchor=(1, 0.5))

    plt.setp(ax2.get_legend().get_texts(), fontsize='12')

    ax1.set_ylim(0, 1.)
    ax2.set_ylim(0, 25)

    common_format(ax2)
    common_format_1(ax1)
    ax2.set_title('length of intervals', y=1.03, size=14)
    ax1.set_title('power of adaptive model', y=1.03, size=14)

    plt.tight_layout(pad=0.2, w_pad=-1.6, h_pad=1.0)
    outfile = os.path.join(outpath, "power_length_comparison_35_real_90.pdf")
    plt.savefig(outfile, format='pdf', bbox_inches='tight')


