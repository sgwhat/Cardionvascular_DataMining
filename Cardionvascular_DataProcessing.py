import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
import seaborn as sns 
import warnings
import pandas_profiling
from pylab import mpl

from sklearn.ensemble import RandomForestClassifier #for the model
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz #plot tree
from sklearn.metrics import roc_curve #for model evaluation
from sklearn.metrics import auc
from sklearn.metrics import classification_report #for model evaluation
from sklearn.metrics import confusion_matrix #for model evaluation
from sklearn.model_selection import train_test_split #for data splitting
import eli5 #for purmutation importance
from eli5.sklearn import PermutationImportance
import shap #for SHAP values
#from pdpbox import pdp, info_plots #for partial plots
np.random.seed(123) #ensure reproducibility
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_curve

#%matplotlib inline

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
#data.describe()
#pandas_profiling.ProfileReport(data)
#sns.countplot(x='cardio', data=data, hue='gender', palette='rainbow')
#plt.show()
'''
dup=data.groupby('gender').size()
bodyhigh = data.groupby('gender')['height'].mean()
print(bodyhigh)
print(dup)
countNoDisease=len(data[data.cardio==0])
countHaveDisease=len(data[data.cardio==1])
countfemale=len(data[data.gender==1])
countmale=len(data[data.gender==2])
print(f'{countNoDisease}',end=',')
print("Sick:{:.2f}%".format((countNoDisease/(len(data.cardio))*100)))
print(f'{countHaveDisease}',end=',')
print("Healthy:{:.2f}%".format((countHaveDisease/(len(data.cardio))*100)))
print(f'{countfemale}',end=',')
print("female_num:{:.2f}%".format((countfemale/(len(data.gender))*100)))
print(f'{countmale}',end=',')
print("male_num:{:.2f}%".format((countmale/(len(data.gender))*100)))
'''

'''
#相关性矩阵
plt.figure(figsize=(15,10))
ax=sns.heatmap(data.corr(), cmap=plt.cm.RdYlBu_r, annot=True, fmt='.2f')
a,b=ax.get_ylim()
ax.set_ylim(a+0.5,b-0.5)
plt.show()
'''

'''
data.loc[data['gender'] == 1, 'gender'] = 'female'
data.loc[data['gender'] == 2, 'gender'] = 'male'

data.loc[data['cp'] == 1, 'cp'] = 'typical'
data.loc[data['cp'] == 2, 'cp'] = 'atypical'
data.loc[data['cp'] == 3, 'cp'] = 'no_pain'
data.loc[data['cp'] == 4, 'cp'] = 'no_feel'

data.loc[data['alco'] == 1, 'alco'] = 'true'
data.loc[data['alco'] == 0, 'alco'] = 'false'

data.loc[data['cholesterol'] == 1, 'cholesterol'] = 'normal'
data.loc[data['cholesterol'] == 2, 'cholesterol'] = 'higher than normal'
data.loc[data['cholesterol'] == 3, 'cholesterol'] = 'much higher than normal'

data.loc[data['smoke'] == 1, 'smoke'] = 'true'
data.loc[data['smoke'] == 0, 'smoke'] = 'false'

data.loc[data['slope'] == 1, 'slope'] = 'up'
data.loc[data['slope'] == 2, 'slope'] = 'flat'
data.loc[data['slope'] == 3, 'slope'] = 'down'

data.loc[data['thal'] == 1, 'thal'] = 'normal'
data.loc[data['thal'] == 2, 'thal'] = 'fixed defect'
data.loc[data['thal'] == 3, 'thal'] = 'reversable defect'
'''

#KNN
y = data['cardio']
x = data.drop(['cardio'], axis = 1)
#x = data.iloc[:, 0:11]
#y = data.iloc[:, 11]
x_train, x_test, y_train, y_test = train_test_split(data.drop('cardio', 1), data['cardio'], test_size = .2, random_state=10) #split the data
#print("Shape of X: {}, Shape of Y: {}".format(x.shape, y.shape))
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 0)

knn_scores = []
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
print("The score for K Neighbors Classifier is {}% with {} nieghbors.".format(knn_scores[7], 19))

kfold = KFold(n_splits=10)  # 将数据集分成十份，进行10折交叉验证: 其中1份作为交叉验证集计算模型准确性，剩余9份作为训练集进行训练
cv_result = cross_val_score(knn_classifier, x, y, cv=kfold)
print("cross val score=", cv_result.mean())

y_predict_knn = knn_classifier.predict(x_test)
f1_score(y_test,y_predict_knn)
print(classification_report(y_test,y_predict_knn))

confusion_matrix = confusion_matrix(y_test,y_predict_knn)
confusion_matrix
print('confusion_matrix:\n' , confusion_matrix)
y_probabilities = knn_classifier.predict_proba(x_test)[:,1]

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

'''
from sklearn.metrics import roc_curve
fprs2,tprs2,thresholds2 = roc_curve(y_test,y_probabilities)
# 此处调用前面的绘制函数
def plot_roc_curve(fprs,tprs):
    plt.figure(figsize=(8,6),dpi=80)
    plt.plot(fprs,tprs)
    plt.plot([0,1],linestyle='--')
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.ylabel('TP rate',fontsize=15)
    plt.xlabel('FP rate',fontsize=15)
    plt.title('ROC曲线',fontsize=17)
    plt.show() 
plot_roc_curve(fprs2,tprs2)

from sklearn.metrics import roc_auc_score  #auc:area under curve
roc_auc_score(y_test,y_probabilities)
print('ROC_AUC_Score:', roc_auc_score)
'''
'''
#Random Forest
train_x, test_x, train_y, test_y = train_test_split(data.drop(columns='cardio'),
                                                    data['cardio'],
                                                    test_size=0.2,
                                                    random_state=10)
train_score = []
test_score = []
for n in range(1, 100):
    model = RandomForestClassifier(max_depth=5,
                                   n_estimators=n,
                                   criterion='gini')
model.fit(train_x, train_y)
train_score.append(model.score(train_x,train_y))
test_score.append(model.score(train_x,train_y))

rf_scores = []
estimators = [10, 100, 200, 500, 1000]
for i in estimators:
    rf_classifier = RandomForestClassifier(n_estimators = i, random_state = 0)
    rf_classifier.fit(X_train, y_train)
    rf_scores.append(rf_classifier.score(X_test, y_test))

colors = rainbow(np.linspace(0, 1, len(estimators)))
plt.bar([i for i in range(len(estimators))], rf_scores, color = colors, width = 0.8)
for i in range(len(estimators)):
    plt.text(i, rf_scores[i], rf_scores[i])
plt.xticks(ticks = [i for i in range(len(estimators))], labels = [str(estimator) for estimator in estimators])
plt.xlabel('Number of estimators')
plt.ylabel('Scores')
plt.title('Random Forest Classifier scores for different number of estimators')
plt.show()
'''
'''
x_axis=[i for i in range(1,100)]

fig,ax=plt.subplots()
ax.plot(x_axis, train_score[:99])
ax.plot(x_axis, test_score[:99],c="r")
plt.xlim([0,100])
plt.ylim([0.0,1.0])
plt.rcParams['font.size'] = 12
plt.xlabel('n_estimators')
plt.ylabel('accuracy')
plt.grid(True)
'''
'''
#kaggle Random Forest examples
#x = data.iloc[:, 0:11]
#y = data.iloc[:, 11]
x_train, x_test, y_train, y_test = train_test_split(data.drop('cardio', 1), data['cardio'], test_size = .2, random_state=10) #split the data
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
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
print('successful')
'''