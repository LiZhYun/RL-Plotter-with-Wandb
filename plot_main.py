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

main_envs = ['HumanoidStandup-v2 17x1', 'Humanoid-v2 17x1']
def plot(df, key='eval_average_episode_rewards'):
    envs = len(main_envs)
    ncol = 2
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
        hue_order = data.algo.unique()

        if idx == 0:
            sns.lineplot(x='step', y=key, data=data, errorbar=('ci', 95), hue='algo',hue_order=hue_order, 
                    palette=COLORS,
                    legend='auto', ax=ax)
        else:
            sns.lineplot(x='step', y=key, data=data, errorbar=('ci', 95), hue='algo', hue_order=hue_order, 
                    palette=COLORS,
                    legend=False, ax=ax)
          
        ax.set_title(scenario)
        ax.set_xlabel('Environment Steps (1e7)')
        ax.set_ylabel('Episode Return')
    plt.tight_layout()
    plt.savefig(f'MuJoCo.pdf')
    plt.show()
    
data_path = './'

df = [pd.read_csv(f'{data_path}/{env}.csv') for env in main_envs]
plot(pd.concat(df))