import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def common_format(ax):
    ax.grid(True, which='both')
    ax.set_xlabel('signal regimes', fontsize=14)
    ax.set_ylabel('time (in seconds)', fontsize=14)
    return ax


def plot_RT(infile_1, infile_2, outpath):

    df_0 = pd.read_csv(infile_1)
    df = df_0[['run_time', 'snr', 'method']]
    df_1 = pd.read_csv(infile_2)
    meth_ =  ['frequentist'] * len(df_1['run_time'])
    df_1['method'] = meth_

    df = df.append(df_1[['run_time', 'snr', 'method']])

    snr_alias = {0.1: 1,
                 0.5: 2,
                 1.0: 3,
                 2.: 4}

    df['snr'] = df['snr'].map(snr_alias)

    sns.set(font_scale=1.8)  # fond size
    sns.set_style("white", {'axes.facecolor': 'white',
                            'axes.grid': True,
                            'axes.linewidth': 2.0,
                            'grid.linestyle': u'--',
                            'grid.linewidth': 4.0,
                            'xtick.major.size': 5.0,
                            })

    cols = ["deepskyblue", "limegreen"]
    order = ["selective", "frequentist"]
    fig = plt.figure(figsize=(12, 4))

    ax = sns.boxplot(x="snr", y="run_time", hue_order=order, hue="method", data=df, palette=cols)

    ax.set(xticklabels=['1', '2', '3', '4'])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    common_format(ax)
    ax.set_title('Distribution of Run-Time', size=14)

    plt.tight_layout(pad=0.2, w_pad=-1.6, h_pad=1.0)
    outfile = os.path.join(outpath, "Run Time.pdf")
    plt.savefig(outfile, format='pdf', bbox_inches='tight')

plot_RT(infile_1="/Users/psnigdha/Research/RadioiBAG/New_Hierarchical_Results_n60_p357/realX_low_PF_inference_35_90_selected.csv",
        infile_2="/Users/psnigdha/Research/RadioiBAG/New_Hierarchical_Results_n60_p357/freq_inf_inference_35_90_selected.csv",
        outpath="/Users/psnigdha/Research/RadioiBAG/Results_New/")