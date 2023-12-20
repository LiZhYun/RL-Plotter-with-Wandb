import pandas as pd
import numpy as np
import scipy as sp
import glob
import os

import matplotlib.pyplot as plt
from matplotlib import cm
plt.style.use('bmh')
import seaborn as sns

plt.rcParams['figure.dpi'] = 400
plt.rcParams['font.size'] = 8
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['legend.loc'] = 'lower right'
plt.rcParams['lines.linewidth'] = 1
COLORS = ['#e41a1c', '#377eb8', '#984ea3', '#ff7f00']

# main_envs = ['academy_3_vs_1_with_keeper', 'academy_counterattack_hard', '5_vs_5']
main_envs = ['academy_3_vs_1_with_keeper', 'academy_counterattack_hard', 'academy_corner']
dir_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(dir_path, 'data')
save_path = os.path.join(dir_path, 'fig')

def plot(df, key='win_rate'):
    envs = len(main_envs)
    ncol = 3
    # assert envs.shape[0] % ncol == 0
    nrow = envs // ncol

    fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))

    for idx, scenario in enumerate(main_envs):
        data = df[df['scenario'] == scenario]
        row = idx // ncol 
        col = idx % ncol
        if nrow == 1:
            ax = axs[col]
        else:
            ax = axs[row, col]

        if idx == 0:
            hue_order = ['BPPO', 'MAPPO', 'ARMAPPO', 'HAPPO']

        if idx == 0:
            sns.lineplot(x='step', y=key, data=data, errorbar=('ci', 95), hue='algo', hue_order=hue_order,
                    palette=COLORS,
                    legend='auto', ax=ax)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles[:], labels=labels[:])
        else:
            sns.lineplot(x='step', y=key, data=data, errorbar=('ci', 95), hue='algo', hue_order=hue_order,
                    palette=COLORS,
                    legend=False, ax=ax)
          
        if scenario == 'academy_3_vs_1_with_keeper':
            ax.set_title('Football 3 vs 1 academy')
        elif scenario == 'academy_counterattack_hard':
            ax.set_title('Football counterattack academy')
        elif scenario == '5_vs_5':
            ax.set_title('Football 5 vs 5')
        elif scenario == 'academy_corner':
            ax.set_title('Football corner academy')
        else:
            raise NameError
        ax.set_xlabel('Environment Steps')
        if idx % ncol == 0:
            ax.set_ylabel('Train Win Rate')
        else:
            ax.set_ylabel('')
    plt.tight_layout()
    plt.savefig(f'{save_path}/Football_train.pdf')
    plt.show()

df = [pd.read_csv(f'{data_path}/{env}' + '_train_' + '.csv') for env in main_envs]
plot(pd.concat(df))