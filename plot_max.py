# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
plt.style.use('bmh')
import seaborn as sns

plt.rcParams['figure.dpi'] = 400
plt.rcParams['font.size'] = 13
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['legend.loc'] = 'lower right'

COLORS = ['#e41a1c', '#377eb8', '#984ea3', '#ff7f00', '#4daf4a',]
# %%
# read data
data_all_pd = pd.read_csv('./ablation_eta_cheetah.csv')
# %%
hue_order =  data_all_pd.eta.unique().sort()

sns.lineplot(x='step', y='max_return', data=data_all_pd, 
             errorbar=('ci', 95), palette=COLORS, hue='eta', hue_order=hue_order, legend='auto'
             ).set(title='HalfCheetah 6x1')
plt.legend(title=r'$\eta$')
plt.xlabel('Environment Steps')
plt.ylabel('Maximal Episode Return')
plt.locator_params(axis='both', nbins=8)
# from matplotlib.ticker import StrMethodFormatter
# plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x: .2f}'))
plt.tight_layout()

current_fig = plt.gcf()
current_fig.savefig('ablation_eta_cheetah.pdf')
# plt.show()
# %%
