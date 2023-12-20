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
COLORS = ['#e41a1c', '#377eb8', '#984ea3', '#ff7f00', '#fbfba1', '#c39797', '#0766AD']
#, '10gen_protoss 10v11', '10gen_terran 10v10', '10gen_terran 10v11'
main_envs = [
            '0','-1'
             ]
dir_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(dir_path, 'data')
save_path = os.path.join(dir_path, 'fig')

algos = ['AengMixer', 'MAPPO', 'HAPPO', 'MAT', 'MAT-Dec']

def plot(df, key='eval_average_episode_rewards'):
    envs = len(main_envs)
    ncol = 2
    # assert envs.shape[0] % ncol == 0
    nrow = envs // ncol

    fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))

    for idx, scenario in enumerate(main_envs):
        data = df[df['scenario'] == int(scenario)]
        row = idx // ncol 
        col = idx % ncol
        if nrow == 1:
            ax = axs[col]
        else:
            ax = axs[row, col]

        if idx == 0:
            hue_order = ['AengMixer', 'MAPPO', 'HAPPO', 'MAT', 'MAT-Dec']
            # hue_order = data.algo.unique()

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
          
        ax.set_title('Predator-Prey Penalty = ' + scenario)
        if (row+1) % (nrow) == 0:
            ax.set_xlabel('Environment Steps')
        else:
            ax.set_xlabel('')
        if idx % ncol == 0:
            ax.set_ylabel('Episode Return')
        else:
            ax.set_ylabel('')
    plt.tight_layout()
    plt.savefig(f'{save_path}/Prey_main.pdf')
    plt.show()

df = [pd.read_csv(f'{data_path}/penalty_{env}.csv') for env in main_envs]
plot(pd.concat(df))