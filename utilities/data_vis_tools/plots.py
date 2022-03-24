#%%

"""
Created on Thu May  7 12:43:20 2020
@author: marko
"""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
#%%
path = '/h/sabrina.atchley/thesis/output_resnet/det_v1n5_c1/'
log =  path + 'log.csv'
#%%
df = pd.read_csv(log, delimiter=';')
save = path  + 'loss.png'
plt.plot(df['epoch'], df['loss'], 'b', linewidth= 2, label = "Training")
plt.plot(df['epoch'], df['val_loss'], 'r', linewidth= 2, label = "Validation")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.grid()
plt.legend()
plt.savefig(save, bbox_inches='tight', dpi=300)
plt.show()

# %%

# %%
save  = path +'accuracy.png'
plt.plot(df['epoch'], df['accuracy'], 'b', linewidth= 2, label = "Training")
plt.plot(df['epoch'], df['val_accuracy'], 'r', linewidth= 2, label = "Validation")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.grid()
plt.legend()
plt.savefig(save, bbox_inches='tight', dpi=300)
plt.show()
# %%

# %%
