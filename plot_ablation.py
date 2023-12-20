# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
plt.style.use('bmh')
import seaborn as sns
COLORS = ['#e41a1c', '#377eb8', '#984ea3', '#ff7f00', '#4daf4a',]
plt.rcParams['figure.dpi'] = 400
plt.rcParams['font.size'] = 13
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['legend.loc'] = 'lower right'

# read data
data_all_pd = pd.read_csv('./ablation_eta_random3.csv')
#%%%
hue_order = data_all_pd.eta.unique().sort()

sns.lineplot(x='step', y='return', data=data_all_pd, 
             errorbar=('ci', 95), palette=COLORS, hue='eta', hue_order=hue_order, legend='auto'
             ).set(title='Random3')
plt.legend(title=r'$\eta$')
plt.xlabel('Environment Steps')
plt.ylabel('Episode Return')
plt.locator_params(axis='both', nbins=8)
plt.tight_layout()
current_fig = plt.gcf()

# %%
current_fig.savefig('ablation_eta_random3_sparse1.pdf')
# plt.show()
# %%
