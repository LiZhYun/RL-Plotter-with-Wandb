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

main_envs = [
             'HalfCheetah-v2 2x3', 'HalfCheetah-v2 3x2', 'HalfCheetah-v2 6x1', 
             'Walker2d-v2 2x3', 'Walker2d-v2 3x2', 'Walker2d-v2 6x1',
             'Humanoid-v2 17x1', 'HumanoidStandup-v2 17x1'
             ]
dir_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(dir_path, 'data')
save_path = os.path.join(dir_path, 'fig')

algos = ['AengMixer', 'MAPPO', 'HAPPO', 'MAT', 'MAT-Dec', 'AIL']

def plot(df, key='eval_average_episode_rewards'):
    envs = len(main_envs)
    ncol = 4
    # assert envs.shape[0] % ncol == 0
    nrow = envs // ncol

    fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))

    for idx, scenario in enumerate(main_envs):
        data = df[df['scenario'] == scenario]
        data = data[data['algo'].isin(algos)]
        row = idx // ncol 
        col = idx % ncol
        if nrow == 1:
            ax = axs[col]
        else:
            ax = axs[row, col]

        if idx == 0:
            hue_order = ['AengMixer', 'MAPPO', 'HAPPO', 'MAT', 'MAT-Dec', 'AIL']
        
        # if scenario == 'HalfCheetah-v2 6x1':
        #     data.loc[data['algo'] == 'ARMAPPO', key] = np.nan
        #     data.dropna()

        if idx == 0:
            sns.lineplot(x='step', y=key, data=data, errorbar=('ci', 95), hue='algo', hue_order=hue_order,
                    palette=COLORS,
                    legend='auto', ax=ax)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles[:], labels=labels[:], loc='upper left')
        else:
            sns.lineplot(x='step', y=key, data=data, errorbar=('ci', 95), hue='algo', hue_order=hue_order,
                    palette=COLORS,
                    legend=False, ax=ax)
          
        ax.set_title(scenario)
        if (row+1) % (nrow) == 0:
            ax.set_xlabel('Environment Steps')
        else:
            ax.set_xlabel('')
        if idx % ncol == 0:
            ax.set_ylabel('Episode Return')
        else:
            ax.set_ylabel('')
    plt.tight_layout()
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,2))
    plt.savefig(f'{save_path}/MuJoCo_abl_app.pdf')
    plt.show()

df = [pd.read_csv(f'{data_path}/{env}.csv') for env in main_envs]
plot(pd.concat(df))