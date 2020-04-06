import numpy as np 
import pandas as pd 

import warnings
from pylab import mpl
import seaborn as sns 
import pandas_profiling

from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.cm import rainbow

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier #for the model

from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve #for model evaluation
from sklearn.tree import export_graphviz #plot tree
from sklearn.model_selection import KFold
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

y = data['cardio']
x = data.drop(['cardio'], axis = 1)
x_train, x_test, y_train, y_test = train_test_split(data.drop('cardio', 1), data['cardio'], test_size = .2, random_state=10) #split the data
model = RandomForestClassifier(max_depth=5)
model.fit(x_train, y_train)

estimator=model.estimators_[1]
features=[i for i in x_train.columns]

y_train_str=y_train.astype('str')
y_train_str[y_train_str=='0']='healthy'
y_train_str[y_train_str=='1']='sick'
y_train_str=y_train_str.values

#Plot Consequent Decision Tree
export_graphviz(estimator, out_file='tree.dot',
				feature_names=features,
				class_names=y_train_str,
				rounded=True, proportion=True,
				label='root',
				precision=2,filled=True)

from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

from IPython.display import Image
Image(filename = 'tree.png')

#Confusion Matrix
y_predict=model.predict(x_test)
y_pred_quant=model.fit(x_train, y_train).predict_proba(x_test)[:, 1]
y_pred_bin = model.predict(x_test)
confusion_matrix = confusion_matrix(y_test, y_pred_bin)
confusion_matrix
print('confusion_matrix:\n' , confusion_matrix)
total=sum(sum(confusion_matrix))
sensitivity=confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[1,0])
print('Sensitivity:', sensitivity)
specificity=confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])
print('Specificity:', specificity)

f1_score(y_test,y_predict)
print(classification_report(y_test,y_predict))
#Plot ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_quant)
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
kfold = KFold(n_splits=10)  # 将数据集分成十份，进行10折交叉验证: 其中1份作为交叉验证集计算模型准确性，剩余9份作为训练集进行训练
cv_result = cross_val_score(model, x, y, cv=kfold)
print("cross val score=", cv_result.mean())
print('successful')

