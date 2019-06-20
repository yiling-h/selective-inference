import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def common_format(ax):
    ax.grid(True, which='both')
    ax.set_xlabel('signal regimes', fontsize=22)
    ax.set_ylabel('', fontsize=22)
    return ax

def plot_discoveries(infile1, infile2, outpath):

    df = pd.read_csv(infile1)
    df = df.append(pd.read_csv(infile2))

    cols = ["deepskyblue", "peachpuff", "salmon", "tomato", "firebrick", "maroon"]
    order = ["selective", "split (50%)", "split (60%)", "split (70%)", "split (80%)", "split (90%)"]

    sns.set(font_scale=1.8)  # fond size
    sns.set_style("white", {'axes.facecolor': 'white',
                            'axes.grid': True,
                            'axes.linewidth': 2.0,
                            'grid.linestyle': u'--',
                            'grid.linewidth': 4.0,
                            'xtick.major.size': 5.0,
                            })

    fig = plt.figure(figsize=(12, 8))
    ax4 = fig.add_subplot(224)
    ax3 = fig.add_subplot(223)
    ax2 = fig.add_subplot(222)
    ax1 = fig.add_subplot(221)

    sns.barplot(x="snr", y="power_screen", hue_order=order,  hue="method", data=df, ax=ax1, palette=cols)
    sns.barplot(x="snr", y="power_total", hue_order=order, hue="method", data=df, ax=ax2, palette=cols)
    sns.barplot(x="snr", y="false_screen", hue_order=order, hue="method", data=df, ax=ax3, palette=cols)
    sns.barplot(x="snr", y="false_total", hue_order=order, hue="method", data=df, ax=ax4, palette=cols)

    ax1.legend_.remove()
    ax2.legend_.remove()
    ax3.legend_.remove()
    ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax1.set_ylim(0, 1.)
    ax2.set_ylim(0, 1.)
    ax3.set_ylim(0, 0.25)
    ax4.set_ylim(0, 0.25)

    common_format(ax1)
    common_format(ax2)
    common_format(ax3)
    common_format(ax4)
    ax1.set_title('power: screening', y=1.03, size=27)
    ax2.set_title('power: total', y=1.03, size=27)
    ax3.set_title('FDP: screening', y=1.03, size=27)
    ax4.set_title('FDP: total', y=1.03, size=27)

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    outfile = os.path.join(outpath, "powerfdr_comparison_35_real_90.pdf")
    plt.savefig(outfile, format='pdf', bbox_inches='tight')

