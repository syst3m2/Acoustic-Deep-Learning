#%%
import pickle
import numpy as np
import scipy.special as sc
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from scipy import interp
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

#%%
with open('/h/sabrina.atchley/thesis/bnnNone.pkl', 'rb') as f:
       bnn=pickle.load(f)

print(bnn.keys())
preds = bnn['preds']
true_labels = bnn['trueLabels']
#%%
print(len(true_labels))

#%%
print(preds.shape)
preds = np.swapaxes(preds,0,1)
print(preds.shape)

#%%
 # True labels are in word form, need to convert to numbers
label_encode = LabelEncoder()
true_labels = label_encode.fit_transform(true_labels)
print(true_labels)
#%%
def BNN_predict(num_classes,to_test):
    pred_vi=np.zeros((len(to_test),num_classes))
    pred_max_p_vi=np.zeros((len(to_test)))
    pred_std_vi=np.zeros((len(to_test)))
    entropy_vi = np.zeros((len(to_test)))

    for i in range(0,len(to_test)):
        preds = to_test[i]
        pred_vi[i]=np.mean(preds,axis=0)#mean over n runs of every proba class
        pred_max_p_vi[i]=np.argmax(np.mean(preds,axis=0))#mean over n runs of every proba class
        pred_std_vi[i]= np.sqrt(np.sum(np.var(preds, axis=0)))
        entropy_vi[i] = -np.sum( pred_vi[i] * np.log2(pred_vi[i] + 1E-14)) #Numerical Stability
    pred_vi_mean_max_p=np.array([pred_vi[i][np.argmax(pred_vi[i])] for i in range(0,len(pred_vi))])
    nll_vi=-np.log(pred_vi_mean_max_p)
    return pred_vi,pred_max_p_vi, pred_vi_mean_max_p, entropy_vi, nll_vi, pred_std_vi

pred_vi, pred_max_p_vi, pred_vi_mean_max_p, entropy_vi, nll_vi, pred_std_vi = BNN_predict(5,preds)
# %%
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, hamming_loss, multilabel_confusion_matrix, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

def classification_metrics(true_labels, pred_labels):

       accuracy = accuracy_score(true_labels, pred_labels)
       print("Accuracy: %.3f" % accuracy)

       print(classification_report(true_labels, pred_labels, target_names=['a', 'b','c','d','e']))
       print("Confusion Matrix:")
       conf_matrix = confusion_matrix(true_labels, pred_labels)
       print(str(conf_matrix))

def classification_metrics_GMI_stra_conv(true_labels, pred_labels_probs):
    pred_labels = pred_labels_probs.argmax(-1)
    accuracy = accuracy_score(true_labels, pred_labels)
    print("Accuracy: %.3f" % accuracy)

    print(classification_report(true_labels, pred_labels, target_names=['a', 'b','c','d','e']))
    print("Confusion Matrix:")
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    print(str(conf_matrix))

    lr_auc = roc_auc_score(true_labels, pred_labels_probs, multi_class='ovr')
    print(' aucROC=%.3f' % (lr_auc))



# %%
print('Report Bayesian')
classification_metrics(true_labels, pred_vi)
# %%
def entropy_hist(entropy, filename):
    counts, bins = np.histogram(entropy, bins=50)
    plt.figure()
    plt.hist(bins[:-1], bins, color='black', weights=counts)
    plt.title(r'Histogram of $H_{pred}$')
    plt.grid()
    plt.savefig(str(filename) + '.png', bbox_inches='tight', dpi=300)

    plt.show()

# %%
print('Report Bayesian AUC')
classification_metrics_GMI_stra_conv(true_labels, pred_vi)

# %%
entropy_hist(entropy_vi, 'Unfiltered')
# %%
mask = entropy_vi <= 0.5
print('Report Bayesian AUC Filtered')
classification_metrics_GMI_stra_conv(true_labels[mask], pred_vi[mask])



# %%
entropy_hist(entropy_vi[mask], 'Filtered Mask = 0.5')




# %%
from matplotlib import rc
import matplotlib as mpl
from pylab import cm
from mpl_toolkits.axes_grid1 import ImageGrid

# Set the global font to be DejaVu Sans, size 10 (or any other sans-serif font of your choice!)
rc('font',**{'family':'sans-serif','sans-serif':['DejaVu Sans'],'size':10})

# Set the font used for MathJax - more on this later
rc('mathtext',**{'default':'regular'})
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1
total_data = len(pred_vi)-1



# %%
f1 = np.zeros(8)
data_percentage = np.zeros(8)
jj=0

# %%
from sklearn.metrics import f1_score
for num in range(25,210,25):
    mask_entropy = pred_std_vi <= (num/100)
    # entropy filtered
    pred_filt = pred_vi[mask_entropy]
    true_labels_filt = true_labels[mask_entropy]
    lr_probs_filt = pred_filt[:, 1]
    pred_class = pred_filt.argmax(-1)
    # calculate scores
    f1[jj] = f1_score(true_labels_filt, pred_class, average='weighted', labels=[0, 1, 2, 3, 4])
    data_percentage[jj] = len(true_labels_filt)/len(true_labels)
    jj+=1
#%%
print(data_percentage)
f1

# %%
fig = plt.figure()
ax1= fig.add_subplot(111)

color = 'tab:red'
ax1.set_xlabel('Ratio of Data Retained Based on $H_{p}$')
ax1.set_ylabel('Weighted F1 Score')
line1 = ax1.plot(data_percentage, f1,marker='v',linestyle='dashed', color=color, label='MC Dropout')
ax1.tick_params(axis='y')
ax1.grid(axis='y', linestyle='--')
ax1.set_xlim(0.0, 1.0)
ax1.set_ylim(0.5, 1.0)
#ax1.text(data_percentage[0]-0.01, 0.926, r'$H_{p}=0.1$')
ax1.annotate(r'$H_{p}=0.25$', xy=(data_percentage[0], f1[0]), xytext=(data_percentage[0]-0.03, 0.905),
            arrowprops=dict(arrowstyle = '->', connectionstyle = 'arc3',facecolor='red'))

ax1.annotate(r'$H_{p}=1.0$', xy=(data_percentage[3], f1[3]), xytext=(data_percentage[3]+0.03, 0.95),
            arrowprops=dict(arrowstyle = '->', connectionstyle = 'arc3',facecolor='red'))
#ax1.text(data_percentage[3]-0.01, 0.916, r'$H_{p}=0.4$')

ax1.annotate(r'$H_{p}=2.0$', xy=(data_percentage[-1], f1[-1]), xytext=(data_percentage[-1]-0.04, 0.885),
            arrowprops=dict(arrowstyle = '->', connectionstyle = 'arc3',facecolor='red'))
data_percentage_flip = [0.04429051, 0.09756507, 0.1672754 , 0.27201931, 0.45782955, 0.68734257, 0.8952351 , 0.98679681]
f1_flip = [0.99472372, 0.98503933, 0.9655309 , 0.92186008, 0.84239308, 0.77215085, 0.71168659, 0.68449541]
line2 = ax1.plot(data_percentage_flip, f1_flip,marker='o',linestyle='dashed', color='tab:green', label='Flipout')
data_percentage_reparam = [0.04101595, 0.08560034, 0.14328296, 0.2393157 , 0.41654072, 0.64746012, 0.87036104, 0.97940806]
f1_reparam = [0.99155123, 0.98244127, 0.96573135, 0.92427505, 0.84163051, 0.76869533, 0.70102944, 0.66498767]
line3 = ax1.plot(data_percentage_reparam, f1_reparam,marker='s',linestyle='dashed', color='tab:blue', label='Reparameterization')
ax1.annotate(r'$H_{p}=0.25$', xy=(data_percentage_reparam[0], f1_reparam[0]), xytext=(data_percentage_reparam[0]-0.04, 0.905),
            arrowprops=dict(arrowstyle = '->', connectionstyle = 'arc3',facecolor='red'))

ax1.annotate(r'$H_{p}=1.0$', xy=(data_percentage_reparam[3], f1_reparam[3]), xytext=(data_percentage_reparam[3]-0.03, 0.81),
            arrowprops=dict(arrowstyle = '->', connectionstyle = 'arc3',facecolor='red'))
ax1.annotate(r'$H_{p}=2.0$', xy=(data_percentage_reparam[-1], f1_reparam[-1]), xytext=(data_percentage_reparam[-1]-0.3, 0.615),
            arrowprops=dict(arrowstyle = '->', connectionstyle = 'arc3',facecolor='red'))
lns = line1+line2+line3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
#plt.grid(which='major', axis='x', linestyle='--')
plt.gcf()
#plt.savefig('./plots/' + 'percentage_of_data_retained_all.eps', bbox_inches='tight', format='eps', dpi=1000)
plt.show()


# %%
