import numpy as np 
import pandas as pd 

import warnings
from pylab import mpl
import seaborn as sns 
import pandas_profiling

from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.cm import rainbow

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve #for model evaluation
from sklearn.tree import export_graphviz #plot tree
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix #for model evaluation
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve #for model evaluation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split #for data splitting

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

#KNN
y = data['cardio']
x = data.drop(['cardio'], axis = 1)
x_train, x_test, y_train, y_test = train_test_split(data.drop('cardio', 1), data['cardio'], test_size = .2, random_state=10) #split the data


knn_scores = [] #accuracy
for k in range(1,21):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    knn_classifier.fit(x_train, y_train)
    knn_scores.append(knn_classifier.score(x_test, y_test))
plt.plot([k for k in range(1, 21)], knn_scores, color = 'red')
for i in range(1,21):
    plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))
plt.xticks([i for i in range(1, 21)])
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Scores')
plt.title('K Neighbors Classifier scores for different K values')
#plt.show()
print("The Accuracy for K Neighbors Classifier is {}% with {} nieghbors.".format(knn_scores[7], 19))

kfold = KFold(n_splits=10)  # 将数据集分成十份，进行10折交叉验证: 其中1份作为交叉验证集计算模型准确性，剩余9份作为训练集进行训练
cv_result = cross_val_score(knn_classifier, x, y, cv=kfold)
print("cv_score=", cv_result.mean())

y_predict_knn = knn_classifier.predict(x_test)
f1_score(y_test,y_predict_knn)
print(classification_report(y_test,y_predict_knn))

confusion_matrix = confusion_matrix(y_test,y_predict_knn)
print(confusion_matrix)
print('confusion_matrix:\n' , confusion_matrix)
y_probabilities = knn_classifier.predict_proba(x_test)[:,1]

knn_result = accuracy_score(y_test,y_predict_knn)
print(knn_result)

recall_knn = confusion_matrix[0][0]/(confusion_matrix[0][0] + confusion_matrix[0][1])
precision_knn = confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[1][1])
print("Recall:",recall_knn,"Precision:",precision_knn)

precisions,recalls,thresholds = precision_recall_curve(y_test,y_probabilities)

plt.plot(thresholds,precisions[:-1])
plt.plot(thresholds,recalls[:-1])
plt.grid()
plt.show()

#Plot ROC
fpr, tpr, thresholds = roc_curve(y_test, y_probabilities)
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
print('successful')