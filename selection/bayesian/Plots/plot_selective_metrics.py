import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# def compute_power(row):
#
#     return row['true_total']/max(1., row['true_screen'])

def common_format(ax):
    ax.grid(True, which='both')
    ax.set_xlabel('signal regimes', fontsize=22)
    ax.set_ylabel('', fontsize=22)
    return ax

def plot_selective_metrics(infile1, infile2, infile3, outpath):

    df = pd.read_csv(infile1)
    df = df.append(pd.read_csv(infile2))
    df = df[df['nactive']>0.]
    method_alias = {"selective": "selective",
                    "split": "split (70%)"}

    df['method'] = df['method'].map(method_alias)

    #df['power'] = df.apply(compute_power, axis=1)
    df = df.append(pd.read_csv(infile3))
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
    #sns.barplot(x="snr", y="power_selective", hue_order=order, hue="method", data=df, ax=ax3, palette=cols)

    ax1.legend_.remove()
    #ax2.legend_.remove()
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax1.set_ylim(0, 1.12)
    ax2.set_ylim(0, 35.)
    #ax3.set_ylim(0, 1.00)

    common_format(ax1)
    common_format(ax2)
    #common_format(ax3)
    ax1.set_title('coverage', y=1.03, size=27)
    ax2.set_title('length', y=1.03, size=27)
    #ax3.set_title('power', y=1.03, size=27)

    ax1.axhline(y=0.9, color='k', linestyle='--', linewidth=2)

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    outfile = os.path.join(outpath, "selective_comparison_35_real_90.pdf")
    plt.savefig(outfile, format='pdf', bbox_inches='tight')

plot_selective_metrics(infile1 = "/Users/psnigdha/Research/RadioiBAG/Results/realX_inference_35_low_90_selected.csv",
                       infile2 = "/Users/psnigdha/Research/RadioiBAG/Results/realX_inference_35_high_90_selected.csv",
                       infile3 = "/Users/psnigdha/Research/RadioiBAG/Results/realX_inference_35_split_90_selected.csv",
                       outpath="/Users/psnigdha/Research/RadioiBAG/Results/")


