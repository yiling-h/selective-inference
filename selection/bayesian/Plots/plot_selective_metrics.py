import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def common_format(ax):
    ax.grid(True, which='both')
    ax.set_xlabel('signal regimes', fontsize=18)
    ax.set_ylabel('', fontsize=20)
    return ax

def plot_selective_metrics(infile1, infile2, infile3, outpath):

    df = pd.read_csv(infile1)
    df = df.append(pd.read_csv(infile2))
    df = df.append(pd.read_csv(infile3))
    df = df[df['nactive'] > 0.]
    df = df[df['snr']<3.]

    snr_alias = {0.1: 1,
                 0.5: 2,
                 1.0: 3,
                 2.5: 4}

    df['snr'] = df['snr'].map(snr_alias)

    cols = ["deepskyblue", "peachpuff", "salmon", "tomato", "firebrick", "maroon"]
    order = ["selective", "split (50%)", "split (60%)", "split (70%)", "split (80%)", "split (90%)"]
    sns.set(font_scale=1.8) # fond size
    sns.set_style("white", {'axes.facecolor': 'white',
                            'axes.grid': True,
                            'axes.linewidth': 2.0,
                            'grid.linestyle': u'--',
                            'grid.linewidth': 4.0,
                            'xtick.major.size': 5.0,
                          })

    fig = plt.figure(figsize=(12, 4))
    #ax3 = fig.add_subplot(133)
    ax2 = fig.add_subplot(132)
    ax1 = fig.add_subplot(131)

    sns.barplot(x="snr", y="coverage", hue_order=order, hue="method", data=df, ax=ax1, palette=cols)
    sns.barplot(x="snr", y="length", hue_order=order, hue="method", data=df, ax=ax2, palette=cols)

    ax1.legend_.remove()
    ax2.legend_.remove()

    ax1.set_ylim(0, 1.12)
    ax2.set_ylim(0, 35.)

    common_format(ax1)
    common_format(ax2)
    ax1.set_title('p/n=5: coverage', y=1.03, size=21)
    ax2.set_title('p/n=5: length', y=1.03, size=21)


    ax1.axhline(y=0.9, color='k', linestyle='--', linewidth=2)

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    outfile = os.path.join(outpath, "selective_low_comparison_power_35_real_90.pdf")
    plt.savefig(outfile, format='pdf', bbox_inches='tight')


