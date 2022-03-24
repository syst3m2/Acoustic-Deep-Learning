import pandas as pd
import os

log_file = '/smallwork/beards/CS4321/AVBW_Team/cs4321-team-sonar-final-project/model_output/current_runs/01_06_2020/output_mel_multilabel_012020_062020_v2_cont_v1/log.csv'

# If continuing to train, then remake the log to fix the epochs since they start over when continuing training
df = pd.read_csv(log_file, delimiter=';')
prev_i = df['epoch'][0]
for i in range(0, len(df['epoch'])):
    if df['epoch'][i] < prev_i:
        df.at[i,'epoch'] = prev_i+1
    prev_i = df['epoch'][i]

df.to_csv(log_file,index=False, sep=';')