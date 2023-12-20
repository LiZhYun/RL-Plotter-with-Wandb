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
plt.rcParams['font.size'] = 12
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['legend.loc'] = 'lower right'
plt.rcParams['lines.linewidth'] = 1
COLORS = ['#e41a1c', '#377eb8', '#984ea3', '#ff7f00', '#fbfba1', '#c39797']

main_envs = ['climbing']
dir_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(dir_path, 'data')
save_path = os.path.join(dir_path, 'fig')

def plot(df, key='eval_average_episode_rewards'):
    envs = len(main_envs)
    ncol = 1
    # assert envs.shape[0] % ncol == 0
    nrow = envs // ncol

    fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))

    for idx, scenario in enumerate(main_envs):
        data = df[df['scenario'] == scenario]
        row = idx // ncol 
        col = idx % ncol
        if ncol == 1:
            if nrow == 1:
                ax = axs
            else:
                ax = axs[row]
        else:
            ax = axs[row, col]

        if idx == 0:
            # hue_order = data.algo.unique()
            hue_order = ['AengMixer', 'MAPPO', 'HAPPO', 'MAVEN', 'MAT-Dec', 'MAT']

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
        if scenario == 'climbing':
            ax.set_title('Climbing Game')
        elif scenario == 'penalty_100':
            ax.set_title('Penalty Game')
        else:
            raise NameError
        
        ax.set_xlabel('Environment Steps')
        if idx % ncol == 0:
            ax.set_ylabel('Episode Return')
        else:
            ax.set_ylabel('')
    plt.tight_layout()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.savefig(f'{save_path}/Matrix.pdf')
    plt.show()

df = [pd.read_csv(f'{data_path}/{env}.csv') for env in main_envs]
plot(pd.concat(df))