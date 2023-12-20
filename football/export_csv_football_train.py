import pandas as pd
import numpy as np
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
import os


def smooth(data, sm=0.2):
  
    i = 1
    moving_averages = []
    moving_averages.append(data[0])

    # Loop through the array elements
    while i < len(data):
    
        # Calculate the exponential
        # average by using the formula
        window_average = (sm*data[i])+(1-sm)*moving_averages[-1]
        
        # Store the cumulative average
        # of current window in moving average list
        moving_averages.append(window_average)
        
        # Shift window to right by one position
        i += 1

    return moving_averages

api = wandb.Api()
entity, project = "zhiyuanli", "Football"

dir_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(dir_path, 'data')

main_envs = ['academy_3_vs_1_with_keeper', 'academy_counterattack_hard', 'academy_corner']
# main_envs = ['academy_3_vs_1_with_keeper', 'academy_counterattack_hard', '5_vs_5', 'academy_corner']
for env in main_envs:
    group = env.split(' ')[0]
    runs = api.runs(entity + "/" + project,
                    filters={"$and": [
                                    {"tags": "exp"}, 
                                    {"tags": "select"}, 
                                    {"state": "finished"}, 
                                    {"group": group}, 
                                    ]}
                                    )

    summary_list = []
    for run in runs:

        history = run.history().dropna()

        if run.name.split('_')[0] == 'temporal':
            algo = 'BPPO'
        elif run.name.split('_')[0] == 'ar':
            algo = 'ARMAPPO'
        elif run.name.split('_')[0] == 'mappo':
            algo = 'MAPPO'
        elif run.name.split('_')[0] == 'happo':
            algo = 'HAPPO'
        else:
            raise NotImplementedError
        
        runs_df = pd.DataFrame({
            "win_rate": history.loc[:, ['win_rate']].values.squeeze(),
            "algo": algo,
            "step": history.loc[:, ["_step"]].values.squeeze(),
            "scenario": env,
            })
        
        summary_list.append(runs_df)

    runs_df = pd.concat(summary_list)

    runs_df.to_csv(f'{data_path}/{env}' + '_train_' + '.csv')