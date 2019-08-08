import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def common_format(ax):
    ax.grid(True, which='both')
    ax.set_xlabel('signal regimes', fontsize=18)
    ax.set_ylabel('', fontsize=20)
    return ax

def compute_true(row):
    return row['true_screen']+(row['false_screen']*row['nactive'])

def plot_discoveries(infile1, infile2,
                     infile3, infile4,
                     infile5, infile6,
                     infile7, infile8,
                     outpath):
    df1 = pd.read_csv(infile1)
    df1 = df1.append(pd.read_csv(infile2))

    df2 = pd.read_csv(infile3)
    df2 = df2.append(pd.read_csv(infile4))

    df3 = pd.read_csv(infile5)
    df3 = df3.append(pd.read_csv(infile6))

    df4 = pd.read_csv(infile7)
    df4 = df4.append(pd.read_csv(infile8))

    cols = ["deepskyblue", "peachpuff", "salmon", "tomato", "firebrick", "maroon"]
    order = ["selective", "split (50%)", "split (60%)", "split (70%)", "split (80%)", "split (90%)"]

    #df['ntrue'] = df.apply(compute_true, axis=1)

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

    #sns.barplot(x="snr", y="ntrue", data=df.loc[df['method'] == "selective"], ax=ax1, palette="Blues_d")
    #sns.barplot(x="snr", y="true_screen", hue_order=order, hue="method", data=df1, ax=ax1, palette=cols)

    sns.barplot(x="snr", y="true_dtotal", hue_order=order, hue="method", data=df1, ax=ax1, palette=cols)
    sns.barplot(x="snr", y="true_dtotal", hue_order=order, hue="method", data=df2, ax=ax2, palette=cols)
    sns.barplot(x="snr", y="true_dtotal", hue_order=order, hue="method", data=df3, ax=ax3, palette=cols)
    sns.barplot(x="snr", y="true_dtotal", hue_order=order, hue="method", data=df4, ax=ax4, palette=cols)

    ax1.legend_.remove()
    ax3.legend_.remove()
    ax4.legend_.remove()
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax1.set_ylim(0, 10)
    ax2.set_ylim(0, 10)
    ax3.set_ylim(0, 10)
    ax4.set_ylim(0, 10)

    common_format(ax1)
    common_format(ax2)
    common_format(ax3)
    common_format(ax4)

    ax1.set_title('p/n=5', y=1.03, size=21)
    ax2.set_title('p/n=2', y=1.03, size=21)
    ax3.set_title('p/n=1', y=1.03, size=21)
    ax4.set_title('p/n=0.5', y=1.03, size=21)

    fig.suptitle('Total discoveries', y=1.01, size=22)

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    outfile = os.path.join(outpath, "discoveries_comparison_35_real_90.pdf")
    plt.savefig(outfile, format='pdf', bbox_inches='tight')



