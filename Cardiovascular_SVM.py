import numpy as np 
import pandas as pd 

import warnings
from pylab import mpl
import seaborn as sns 
import pandas_profiling

from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.cm import rainbow

from sklearn.svm import SVC

from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve #for model evaluation
from sklearn.tree import export_graphviz #plot tree
from sklearn.metrics import confusion_matrix #for model evaluation
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve
from sklearn.metrics import precision_recall_curve #for model evaluation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split #for data splitting
from sklearn.model_selection import GridSearchCV

np.random.seed(123) #ensure reproducibility

warnings.filterwarnings('ignore')

data=pd.read_csv('/usr/local/dataset/cardio_train.csv', 
                        sep = ';' 
                        #names = ['age','gender','height','weight',
                                 #'ap_hi','ap_lo','cholesterol','glu',
                                 #'smoke','alco','active',
                                 #'cardio']
                                 )

data.drop(columns=['id'], inplace=True)
#data.head()
data.info()
print(data.groupby("cardio").size())
data.dtypes
print(data.dtypes)

#拆分数据集
y = data['cardio']
X = data.drop(['cardio'], axis = 1)
print("Shape of X: {0}; positive example: {1}; negative: {2}".format(X.shape, y[y==1].shape[0], y[y==0].shape[0]))  # 查看数据的形状和类别分布
X_train, X_test, y_train, y_test = train_test_split(data.drop('cardio', 1), data['cardio'], test_size = 0.2, random_state=10) #split the data

#训练模型: RBF核函数
clf = SVC(C=1.0, kernel="rbf", gamma=0.1)
clf.fit(X_train, y_train)
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
print("train score: {0}; test score: {1}".format(train_score, test_score))

y_pred_svc =clf.predict(X_test)
cm_svc = confusion_matrix(y_test, y_pred_svc)
print(cm_svc)

svc_result = accuracy_score(y_test,y_pred_svc)
print("Accuracy:",svc_result)

f1_score(y_test,y_pred_svc)
print(classification_report(y_test,y_pred_svc))

recall = cm_svc[0][0]/(cm_svc[0][0] + cm_svc[0][1])
precision = cm_svc[0][0]/(cm_svc[0][0]+cm_svc[1][1])
print("Recall:",recall, "Precision:",precision)

f1_score(y_test,y_pred_svc)
print(classification_report(y_test,y_pred_svc))

#Plot ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_svc)
fig, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
#plt.show()

#Compute AUC
roc_auc = auc(fpr, tpr)
print('ROC_AUC_Score:', roc_auc)
'''
#模型调优
gammas = np.linspace(0, 0.0003, 30)
param_grid = {"gamma": gammas}
clf = GridSearchCV(SVC(), param_grid, cv=5)
clf.fit(X, y)
print("best param: {0}\n best score: {1}".format(clf.best_params_, clf.best_score_))

#训练模型: 二阶多项式核函数
clf = SVC(C=1.0, kernel="poly", degree=2)
clf.fit(X_train, y_train)
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
print("train score: {0}; test score: {1}".format(train_score, test_score))


#绘制学习曲线
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
title = "Learning Curves with degree={0}"
degrees = [1, 2]
plt.figure(figsize=(12, 4), dpi=144)
for i in range(len(degrees)):
    plt.subplot(1, len(degrees), i+1)
    plot_learning_curve(plt, SVC(C=1.0, kernel="poly", degree=degrees[i]), title.format(degrees[i]), X, y, ylim=(0.8, 1.01), cv=cv)
plt.show()
'''