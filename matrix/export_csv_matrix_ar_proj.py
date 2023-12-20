import pandas as pd
import numpy as np
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
import os


def smooth(data, sm=1):
  
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
entity, project = "zhiyuanli", "matrix"

dir_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(dir_path, 'data')

main_envs = ['climbing', 'penalty_100']
for env in main_envs:
    group = env
    runs = api.runs(entity + "/" + project,
                    filters={"$and": [
                                    {"tags": "exp"}, 
                                    {"state": "finished"}, 
                                    {"group": group}, 
                                    {"config.algorithm_name": 'ar'},
                                    ]}
                                    )

    summary_list = []
    for run in runs:

        history = run.history().dropna()

        if 'ar_proj' in run.tags:
            algo = 'ARMAPPO w/ PROJ'
        else:
            algo = 'ARMAPPO '
        
        runs_df = pd.DataFrame({
            "eval_average_episode_rewards": history.loc[:, ['agent0/average_episode_rewards_by_eplength']].values.squeeze(),
            "algo": algo,
            "step": history.loc[:, ["_step"]].values.squeeze(),
            "scenario": env,
            })
        
        summary_list.append(runs_df)

    runs_df = pd.concat(summary_list)

    runs_df.to_csv(f'{data_path}/{env}' + '_ar_proj_' + '.csv')