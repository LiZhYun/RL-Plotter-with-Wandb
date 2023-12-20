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
entity, project = "zhiyuanli", "StarCraft2v2"

dir_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(dir_path, 'data')

main_envs = ['10gen_protoss 5v5', '10gen_protoss 10v10', '10gen_protoss 10v11',
             '10gen_zerg 5v5', '10gen_zerg 10v10', '10gen_zerg 10v11',
             '10gen_terran 5v5', '10gen_terran 10v10', '10gen_terran 10v11',
             ]
for env in main_envs:
    group = env.split(' ')[0]
    units = env.split(' ')[1]
    runs = api.runs(entity + "/" + project,
                    filters={"$and": [
                                    {"tags": "iclr24_select"}, 
                                    # {"tags": "select"}, 
                                    # {"state": "finished"}, 
                                    {"group": group}, 
                                    {"config.units": units},
                                    ]}
                                    )

    summary_list = []
    for run in runs:

        history = run.history().dropna()

        if run.name.split('_')[0] == 'temporal':
            algo = 'AengMixer'
        elif run.name.split('_')[0] == 'mappo':
            algo = 'MAPPO'
        elif run.name.split('_')[0] == 'happo':
            algo = 'HAPPO'
        elif run.name.split('_')[1] == 'dec':
            algo = 'MAT-Dec'
        elif run.name.split('_')[0] == 'mat':
            algo = 'MAT'
        elif run.name.split('_')[0] == 'maven':
            algo = 'MAVEN'
        else:
            raise NotImplementedError
        if history.empty:
            continue
        runs_df = pd.DataFrame({
            "eval_win_rate": smooth(history.loc[:, ['eval_win_rate']].values.squeeze()),
            "algo": algo,
            "step": history.loc[:, ["_step"]].values.squeeze(),
            "scenario": env,
            })
        
        summary_list.append(runs_df)

    runs_df = pd.concat(summary_list)

    runs_df.to_csv(f'{data_path}/{env}.csv')