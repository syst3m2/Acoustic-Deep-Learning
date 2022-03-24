#%%


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
#%%
path = '/smallwork/beards/CS4321/AVBW_Team/cs4321-team-sonar-final-project/outputs/model_training/current_runs/multilabel_2020_week_split/'
log =  path + 'log.csv'
#%%
df = pd.read_csv(log, delimiter=';')
save = path  + 'loss.png'
plt.plot(df['epoch'], df['loss'], 'b', linewidth= 2, label = "Training Loss")
plt.plot(df['epoch'], df['val_loss'], 'r', linewidth= 2, label = "Validation Loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.grid()
plt.legend(loc='center right')
plt.savefig(save, bbox_inches='tight', dpi=300)
plt.show()

# %%

# %%
save  = path +'accuracy.png'
plt.plot(df['epoch'], df['accuracy'], 'g', linewidth= 2, label = "Training Accuracy")
plt.plot(df['epoch'], df['val_accuracy'], 'purple', linewidth= 2, label = "Validation Accuracy")
plt.xlabel("epoch")
plt.ylabel("loss / accuracy")
plt.grid()
plt.legend(loc='center right')
plt.savefig(save, bbox_inches='tight', dpi=300)
plt.show()
# %%

# %%
