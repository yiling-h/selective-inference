import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def common_format(ax):
    ax.grid(True, which='both')
    ax.set_xlabel('signal regimes', fontsize=22)
    ax.set_ylabel('', fontsize=22)
    return ax

def plot_discoveries(infile, outpath):

    df = pd.read_csv(infile)
    cols = ["#3498db", "#FF8C00"]
    order = ["selective", "split"]
    sns.set(font_scale=1.8)  # fond size
    sns.set_style("white", {'axes.facecolor': 'white',
                            'axes.grid': True,
                            'axes.linewidth': 2.0,
                            'grid.linestyle': u'--',
                            'grid.linewidth': 4.0,
                            'xtick.major.size': 5.0,
                            })

    fig = plt.figure(figsize=(12, 5))
    ax3 = fig.add_subplot(133)
    ax2 = fig.add_subplot(132)
    ax1 = fig.add_subplot(131)

    sns.barplot(x="snr", y="ntrue", data=df.loc[df['method'] == "selective"], ax=ax1, palette="Blues_d")
    sns.pointplot(x="snr", y="true_screen", hue_order=order, markers='o', hue="method", data=df, ax=ax2, palette=cols)
    sns.pointplot(x="snr", y="true_total", hue_order=order, markers='o', hue="method", data=df, ax=ax3, palette=cols)

    ax2.legend_.remove()
    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax1.set_ylim(0, 24)
    ax2.set_ylim(0, 8)
    ax3.set_ylim(0, 4)

    common_format(ax1)
    common_format(ax2)
    common_format(ax3)
    ax1.set_title('signals', y=1.03, size=27)
    ax2.set_title('screened signals', y=1.03, size=27)
    ax3.set_title('discovered signals', y=1.03, size=27)

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    outfile = os.path.join(outpath, "discoveries_comparison.pdf")
    plt.savefig(outfile, format='pdf', bbox_inches='tight')

plot_discoveries(infile = "/Users/psnigdha/Research/RadioiBAG/Results/dims_65_350_inference_selected_rho_0.3.csv", outpath="/Users/psnigdha/Research/RadioiBAG/Results/")

