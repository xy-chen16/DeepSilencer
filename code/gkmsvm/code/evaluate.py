import numpy as np 
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn import metrics

# load the ouput from gkmSVM
path_output = './saved_data/output.txt'
data = np.loadtxt(path_output,dtype=str,delimiter="\t")

# predict probability
prob = data[:,1]
prob = prob.astype(float)
num = len(prob)

# generate label
label = np.zeros((num,1))
label[0:int(num/2)] = 1

# calculate ROAUC and PRAUC
fpr,tpr,thresholds=roc_curve(label,prob)
precision, recall, thresholds = precision_recall_curve(label, prob)
auc=roc_auc_score(label,prob) 
print("AUC(ROC):" + str(auc))
area = metrics.auc(recall, precision)
print("AUC(PRC):" + str(area))