#%%
import pickle
import numpy as np
import scipy.special as sc
import pandas as pd
from scipy import interp
from itertools import cycle
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, hamming_loss, multilabel_confusion_matrix, roc_auc_score, average_precision_score, roc_curve, auc, f1_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, label_binarize
from matplotlib import rc
import matplotlib as mpl
from pylab import cm
from mpl_toolkits.axes_grid1 import ImageGrid
import copy
#%%
with open('bnn.pkl', 'rb') as f:
       bnn=pickle.load(f)

print(bnn.keys())
preds = bnn['preds']
true_labels = bnn['trueLabels']

#%%
 # True labels are in word form, need to convert to numbers
 # This is true of the way Andrew was doing the data pipeline, Nick's new pipleline already does this step
mlb = MultiLabelBinarizer()
true_multi = mlb.fit_transform(true_labels)

#%%
print(preds.shape)
preds = np.swapaxes(preds,0,1)
print(preds.shape)



#%%
def BNN_predict(num_classes, to_test):
    pred_vi=np.zeros((len(to_test),num_classes))
    pred_max_p_vi=np.zeros((len(to_test)))
    pred_std_vi=np.zeros((len(to_test)))
    entropy_vi = np.zeros((len(to_test)))
    normal_entropy_vi = np.zeros((len(to_test)))
    var=  np.zeros((len(to_test)))
    for i in range(0,len(to_test)):
        preds = to_test[i]
        pred_vi[i]=np.mean(preds,axis=0)#mean over n runs of every proba class
        pred_max_p_vi[i]=np.argmax(np.mean(preds,axis=0))#mean over n runs of every proba class
        pred_std_vi[i]= np.sqrt(np.sum(np.var(preds, axis=0)))
        var[i] =  np.sum(np.var(preds, axis=0))
        entropy_vi[i] = -np.sum( pred_vi[i] * np.log2(pred_vi[i] + 1E-14)) #Numerical Stability
        normal_entropy_vi[i] = entropy_vi[i]/np.log2(2^num_classes)
    pred_vi_mean_max_p=np.array([pred_vi[i][np.argmax(pred_vi[i])] for i in range(0,len(pred_vi))])
    nll_vi=-np.log(pred_vi_mean_max_p)
    return pred_vi,pred_max_p_vi, pred_vi_mean_max_p, entropy_vi, nll_vi, pred_std_vi, var, normal_entropy_vi

pred_vi, pred_max_p_vi, pred_vi_mean_max_p, entropy_vi, nll_vi, pred_std_vi, var, normal_entropy = BNN_predict(5,preds)

#%%
#This is the logic that looks at edge cases involving Class E, and when no class reaches 0.5 probability before
# making the final predictions
pred_vi_working = copy.deepcopy(pred_vi)
for item in pred_vi_working:
    if item[4] >= 0.5:
        #if predictive probablitly of class E is >= to .5, get the index (which will correspond to class)
        # of the highest predictive probability
        k = np.argmax(item)
        if k == 4:
            #if the highest probability is class E, predict class E and nothing else
            for i in range(len(item)):
                item[i] = 0
            item[4] = 1
        else:
            #if class E is not the highest, don't predict it, predict everything else that is over 0.5
            item[4] = 0
            for i in range(len(item)):
                if item[i] >=0.5:
                    item[i] = 1
                else:
                    item[i] = 0
    elif  max(item) < 0.5:
        #if no class has a predictive probability over 0.5, just predict the highest
        k = np.argmax(item)
        for i in range(len(item)):
            item[i] = 0
        item[k] = 1
    else:
        for i in range(len(item)):
            if item[i] >=0.5:
                item[i] = 1
            else:
                item[i] = 0

print(pred_vi_working[0:5])

# %%
def classification_metrics_ml(true_labels, pred_labels):
    accuracy = accuracy_score(true_labels, pred_labels)
    print("Accuracy: %.3f" % accuracy)

    print(classification_report(true_labels, pred_labels, target_names=['a', 'b','c','d','e']))
    print("Confusion Matrix:")
    conf_matrix = multilabel_confusion_matrix(true_labels, pred_labels)

    print(str(conf_matrix))

    lr_auc = roc_auc_score(true_labels, pred_labels, multi_class='ovr')
    print(' aucROC=%.3f' % (lr_auc))

    hloss = hamming_loss(true_labels, pred_labels)
    print("Hamming loss: %.3f" % hloss)

    return conf_matrix, accuracy, lr_auc, hloss

# %%
def entropy_hist(entropy, filename):
    #creates entropy histogram
    counts, bins = np.histogram(entropy, bins=50)
    plt.figure()
    plt.hist(bins[:-1], bins, color='black', weights=counts)
    plt.title(r'Histogram of $H_{pred}$')
    plt.grid()
    plt.savefig('./plots/entropy_hist_' + str(filename) + '.png', bbox_inches='tight', dpi=300)

    plt.show()
#%%
def var_hist(var, filename):
    #creates total variance histogram
    counts, bins = np.histogram(var, bins=25)
    plt.figure()
    plt.hist(bins[:-1], bins, color='black', weights=counts)
    plt.title(r'Histogram of $var_{pred}$')
    plt.grid()
    plt.savefig('./plots/var_hist_' + str(filename) + '.png', bbox_inches='tight', dpi=300)

    plt.show()

#%%
def normal_entropy_hist(entropy, filename):
    #creates normalized entropy histogram
    counts, bins = np.histogram(entropy, bins=50)
    plt.figure()
    plt.hist(bins[:-1], bins, color='black', weights=counts)
    plt.title(r'Histogram of $H^*_{pred}$')
    plt.grid()
    plt.savefig('./plots/normal_entropy_hist_' + str(filename) + '.png', bbox_inches='tight', dpi=300)

    plt.show()

# %%
entropy_hist(entropy_vi, 'Unfiltered')

#%%
normal_entropy_hist(normal_entropy, 'Normalized')
#%%
var_hist(var, 'Unfiltered')
# %%
mask = entropy_vi <= 0.5
#%%
print(pred_vi_working[mask])
print(true_multi[mask])
#%%
print('Report Bayesian AUC Filtered')
classification_metrics_ml(true_multi[mask], pred_vi_working[mask])


# %%
entropy_hist(entropy_vi[mask], 'Filtered Mask = 0.5')


#%%

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
for num in range(125,1100,125):
    mask_entropy = normal_entropy <= (num/1000)
    print(num/1000)
    # entropy filtered
    pred_filt = pred_vi_working[mask_entropy]
    true_labels_filt = true_multi[mask_entropy]
    # calculate scores
    f1[jj] = f1_score(true_labels_filt, pred_filt, average='weighted', labels=[0, 1, 2, 3, 4])
    data_percentage[jj] = len(true_labels_filt)/51372
    jj+=1
#%%
#Display arrays so you can copy/past into text document to carry over to the one you want to use to plot the graph
print(data_percentage)
print(f1)

# %%
fig = plt.figure()
ax1= fig.add_subplot(111)

color = 'tab:red'
ax1.set_xlabel('Ratio of Data Retained Based on $H^*_{p}$')
ax1.set_ylabel('Weighted F1 Score')
line1 = ax1.plot(data_percentage[:-1], f1[:-1],marker='v',linestyle='dashed', color=color, label='Flipout Multi-label')
ax1.tick_params(axis='y')
ax1.grid(axis='y', linestyle='--')
ax1.set_xlim(0.3, 1.02)
ax1.set_ylim(0.5, 1.02)
#ax1.text(data_percentage[0]-0.01, 0.926, r'$H_{p}=0.1$')

data_percentage_flip_mc = [0.45014694, 0.57141058, 0.80782955, 0.92063392, 0.99019731, 0.99962217, 1.0]
f1_flip_mc = [0.9849673, 0.96358224, 0.86349901, 0.82471229, 0.79583833, 0.79191056, 0.79181626]
line2 = ax1.plot(data_percentage_flip_mc, f1_flip_mc,marker='o',linestyle='dashed', color='tab:green', label='Flipout Multi-class')
data_percentage_drop_mc = [0.46446264, 0.58776238, 0.79737615, 0.92611251, 0.99233837, 0.99974811, 1.0]
f1_drop_mc = [0.99669854, 0.98703427, 0.90078556, 0.85580929, 0.82652786, 0.82331957, 0.82316636]
line3 = ax1.plot(data_percentage_drop_mc, f1_drop_mc,marker='s',linestyle='dashed', color='tab:blue', label='Dropout Multi-class')

ax1.annotate(r'$H^*_{p}=0.125$', xy=(data_percentage_drop_mc[0], f1_drop_mc[0]), xytext=(data_percentage_drop_mc[0]-0.1, 0.90),
            arrowprops=dict(arrowstyle = '->', connectionstyle = 'arc3',facecolor='red'))

ax1.annotate(r'$H^*_{p}=0.5$', xy=(data_percentage_drop_mc[3], f1_drop_mc[3]), xytext=(data_percentage_drop_mc[3]-0.18, 0.97),
            arrowprops=dict(arrowstyle = '->', connectionstyle = 'arc3',facecolor='red'))
# # ax1.text(data_percentage[3]-0.01, 0.916, r'$H_{p}=0.4$')

ax1.annotate(r'$H^*_{p}=1.0$', xy=(data_percentage_drop_mc[-1], f1_drop_mc[-1]), xytext=(data_percentage_drop_mc[-1] - 0.13, 0.92),
             arrowprops=dict(arrowstyle = '->', connectionstyle = 'arc3',facecolor='red'))
data_percentage_drop_ml = [0.6175021, 0.75999519, 0.86392595, 0.94819089, 0.98687342, 0.99826902, 1.0]
f1_drop_ml = [0.89741596, 0.89206708, 0.87746858, 0.85551307, 0.84411251, 0.83996867, 0.83920584]
line4 = ax1.plot(data_percentage_drop_ml, f1_drop_ml,marker='P',linestyle='dashed', color='tab:orange', label='Dropout Multi-label')
ax1.annotate(r'$H^*_{p}=0.125$', xy=(data_percentage[0], f1[0]), xytext=(data_percentage[0]-.09, 0.81),
            arrowprops=dict(arrowstyle = '->', connectionstyle = 'arc3',facecolor='red'))
ax1.annotate(r'$H^*_{p}=0.5$', xy=(data_percentage[3], f1[3]), xytext=(data_percentage[3]-0.26, 0.75),
             arrowprops=dict(arrowstyle = '->', connectionstyle = 'arc3',facecolor='red'))
ax1.annotate(r'$H^*_{p}=1.0$', xy=(data_percentage[-1], f1[-1]), xytext=(data_percentage[-1]-0.10, 0.72),
             arrowprops=dict(arrowstyle = '->', connectionstyle = 'arc3',facecolor='red'))
lns = line1+line2+line3+line4
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
#plt.grid(which='major', axis='x', linestyle='--')
plt.gcf()
plt.savefig('./plots/' + 'percentage_of_data_retained_normal.eps', bbox_inches='tight', format='eps', dpi=1000)
plt.savefig('./plots/' + 'percentage_of_data_retained_normal.jpg', bbox_inches='tight', format='jpg', dpi=1000)
plt.show()

# %%
