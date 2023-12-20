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
entity, project = "zhiyuanli", "predator_prey"

dir_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(dir_path, 'data')

main_envs = [
            '0','-1'
             ]
for env in main_envs:
    # group = env.split(' ')[0]
    # units = env.split(' ')[1]
    runs = api.runs(entity + "/" + project,
                    filters={"$and": [
                                    {"tags": "iclr24_select"}, 
                                    # {"tags": "select"}, 
                                    # {"state": "finished"}, 
                                    # {"group": group}, 
                                    {"config.penalty": int(env)},
                                    ]}
                                    )

    summary_list = []
    for run in runs:

        history = run.history().dropna()

        if run.config['algorithm_name'] == 'temporal':
            algo = 'AengMixer'
        elif run.config['algorithm_name'] == 'mappo':
            algo = 'MAPPO'
        elif run.config['algorithm_name'] == 'happo':
            algo = 'HAPPO'
        elif run.config['algorithm_name'] == 'mat_dec':
            algo = 'MAT-Dec'
        elif run.config['algorithm_name'] == 'mat':
            algo = 'MAT'
        # elif run.config['algorithm_name'] == 'maven':
        #     algo = 'MAVEN'
        # # elif run.config['algorithm_name'] == 'mqmix':
        # #     algo = 'MACPF'
        # elif run.config['algorithm_name'] == 'single':
        #     algo = 'MACPF'
        # elif run.config['algorithm_name'] == 'full':
        #     algo = 'MAPPO_FULL'
        else:
            continue
        if history.empty:
            continue
        runs_df = pd.DataFrame({
            "eval_average_episode_rewards": smooth(history.loc[:, ['eval_average_episode_rewards']].values.squeeze()),
            "algo": algo,
            "step": history.loc[:, ["_step"]].values.squeeze(),
            "scenario": env,
            })
        
        summary_list.append(runs_df)

    runs_df = pd.concat(summary_list)

    runs_df.to_csv(f'{data_path}/penalty_{env}.csv')