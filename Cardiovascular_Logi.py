import numpy as np
import numpy as np 
import pandas as pd 

import warnings
from pylab import mpl
import seaborn as sns 
import pandas_profiling
import warnings

from sklearn.metrics import auc
from sklearn.metrics import roc_curve #for model evaluation
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.cm import rainbow
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve #for model evaluation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix #for model evaluation
from sklearn.tree import export_graphviz #plot tree
from sklearn.model_selection import KFold

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
X = data.drop(['cardio'], axis = 1)
print("Shape of X: {0}; positive example: {1}; negative: {2}".format(X.shape, y[y==1].shape[0], y[y==0].shape[0]))  # 查看数据的形状和类别分布
X_train, X_test, y_train, y_test = train_test_split(data.drop('cardio', 1), data['cardio'], test_size = .2, random_state=10) #split the data

model = LogisticRegression()
model.fit(X_train, y_train)
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print("train score: {train_score:.6f}; test score: {test_score:.6f}".format(train_score=train_score, test_score=test_score))

#模型预测
y_pred = model.predict(X_test)
print("matchs: {0}/{1}".format(np.equal(y_pred, y_test).shape[0], y_test.shape[0]))
lr_result = accuracy_score(y_test,y_pred)
print("Accuracy:", lr_result)

# 构建多项式模型
def polynomial_model(degree=1, **kwarg):
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    logistic_regression = LogisticRegression(**kwarg)
    pipeline = Pipeline([("polynomial_feature", polynomial_features), ("logistic_regression", logistic_regression)])
    return pipeline

model = polynomial_model(degree=2, penalty="l1")
model.fit(X_train, y_train)

train_score_v2 = model.score(X_train, y_train)
test_score_v2 = model.score(X_test, y_test)
print("train_score_v2: {:.6f}, test_score_v2: {:.6f}".format(train_score_v2, test_score_v2))

logistic_regression = model.named_steps["logistic_regression"]
print("model_parameters shape: {0}; count of non-zero element: {1}".format(logistic_regression.coef_.shape,
                                                                          np.count_nonzero(logistic_regression.coef_)))
y_predict = model.predict(X_test)
print("matchs_v2: {0}/{1}".format(np.equal(y_predict, y_test).shape[0], y_test.shape[0]))
lr_result = accuracy_score(y_test,y_predict)
print("Accuracy_v2:", lr_result)

f1_score(y_test,y_predict)
print(classification_report(y_test,y_predict))

confusion_matrix = confusion_matrix(y_test,y_predict)
print(confusion_matrix)
print('confusion_matrix:\n' , confusion_matrix)
y_predict = model.predict_proba(X_test)[:,1]

recall = confusion_matrix[0][0]/(confusion_matrix[0][0] + confusion_matrix[0][1])
precision = confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[1][1])
print("Recall:",recall,"Precision:",precision)

precisions,recalls,thresholds = precision_recall_curve(y_test,y_predict)

plt.plot(thresholds,precisions[:-1])
plt.plot(thresholds,recalls[:-1])
plt.grid()
plt.show()

#Plot ROC
fpr, tpr, thresholds = roc_curve(y_test, y_predict)
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





