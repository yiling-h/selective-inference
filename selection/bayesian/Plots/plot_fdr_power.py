import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def common_format(ax):
    ax.grid(True, which='both')
    ax.set_xlabel('signal regimes', fontsize=18)
    ax.set_ylabel('', fontsize=20)
    return ax

def compute_power(row):
    if row['power_screen']>1.:
        return 1
    else:
        return row['power_screen']

def plot_fdr_power(infile1, infile2, outpath):

    df = pd.read_csv(infile1)
    df['power_screen_cap'] = df.apply(compute_power, axis=1)
    df_1 = pd.read_csv(infile2)
    df_1['power_screen_cap'] = df_1.apply(compute_power, axis=1)
    df = df.append(df_1)
    print("check ", max(df['power_screen_cap']), min(df['power_screen_cap']))

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

    sns.barplot(x="snr", y="power_screen_cap", hue_order=order,  hue="method", data=df, ax=ax1, palette=cols)
    sns.barplot(x="snr", y="power_total", hue_order=order, hue="method", data=df, ax=ax2, palette=cols)
    sns.barplot(x="snr", y="false_screen", hue_order=order, hue="method", data=df, ax=ax3, palette=cols)
    sns.barplot(x="snr", y="false_total", hue_order=order, hue="method", data=df, ax=ax4, palette=cols)

    ax1.legend_.remove()
    ax4.legend_.remove()
    ax3.legend_.remove()
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax1.set_ylim(0, 1.10)
    ax2.set_ylim(0, 1.10)
    ax3.set_ylim(0, 1.)
    ax4.set_ylim(0, 1.)

    common_format(ax1)
    common_format(ax2)
    common_format(ax3)
    common_format(ax4)
    ax1.set_title('p/n=2: screening power', y=1.03, size=21)
    ax2.set_title('p/n=2: total power', y=1.03, size=21)
    ax3.set_title('p/n=2: screening FDP', y=1.03, size=21)
    ax4.set_title('p/n=2: total FDP', y=1.03, size=21)

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    outfile = os.path.join(outpath, "powerfdr_comparison_35_real_90.pdf")
    plt.savefig(outfile, format='pdf', bbox_inches='tight')


