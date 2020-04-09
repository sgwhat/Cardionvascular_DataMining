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

y = data['cardio']
X = data.drop(['cardio'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(data.drop('cardio', 1), data['cardio'], test_size = .2, random_state=10) #split the data
print("X_train shape:", X_train.shape, "X_test shape:", X_test.shape)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print("train score:{0:.3f}; test score:{1:.3f}".format(train_score, test_score))

# 优化模型参数：max_depth
def cv_score(d):
    model = DecisionTreeClassifier(max_depth=d)
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    return (train_score, test_score)

depths = range(2, 15)
scores = [cv_score(d) for d in depths]
train_scores = [s[0] for s in scores]
cv_scores = [s[1] for s in scores]

# 找出交叉验证集评分最高的模型参数
best_score_index = np.argmax(cv_scores)
best_score = cv_scores[best_score_index]
best_param = depths[best_score_index]   # 找出对应的参数
print("best param: {0}; best score: {1:.3f}".format(best_param, best_score))

plt.figure(figsize=(6, 4), dpi=200)
plt.grid()
plt.xlabel("Max depth of Decision Tree")
plt.ylabel("score")
plt.plot(depths, cv_scores, ".g--", label="cross validation score")
plt.plot(depths, train_scores, ".r--", label="training score")
plt.legend()
#plt.show()

# 优化模型参数：在criterion="gini"下的min_impurity_split
def cv_score(val):
    """
    在不同depth值下，train_score和test_score的值
    :param d: max_depth值
    :return: (train_score, test_score)
    """
    model = DecisionTreeClassifier(criterion="gini", min_impurity_split=val)
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    return (train_score, test_score)


# 指定参数的范围，训练模型计算得分
values = np.linspace(0, 0.5, 50)
scores = [cv_score(v) for v in values]
train_scores = [s[0] for s in scores]
cv_scores = [s[1] for s in scores]

# 找出交叉验证集评分最高的模型参数
best_score_index = np.argmax(cv_scores)
best_score = cv_scores[best_score_index]
best_param = values[best_score_index]   # 找出对应的参数
print("best param: {0}; best score: {1:.3f}".format(best_param, best_score))

# 画出模型参数与模型评分的关系
plt.figure(figsize=(6, 4), dpi=200)
plt.grid()
plt.xlabel("Min_impurity_split of Decision Tree")
plt.ylabel("score")
plt.plot(values, cv_scores, ".g--", label="cross validation score")
plt.plot(values, train_scores, ".r--", label="training score")
plt.legend()
#plt.show()

# 参数
entropy_thresholds = np.linspace(0, 1, 50)
gini_thresholds = np.linspace(0, 0.5, 50)

# 设置参数矩阵
param_grid = [{"criterion": ["entropy"], "min_impurity_split": entropy_thresholds},
              {"criterion": ["gini"], "min_impurity_split": gini_thresholds},
              {"max_depth": range(2, 10)},
              {"min_samples_split": range(2, 30, 2)}]

model = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
model.fit(X, y)
print("best param: {0} \nbest score: {1}".format(model.best_params_, model.best_score_))

#生成决策树图形
model = DecisionTreeClassifier(criterion='entropy', min_impurity_split=0.5306122448979591)
model.fit(X_train, y_train)
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print('train score: {0:.3f}; test score: {1:.3f}'.format(train_score, test_score))

y_pred = model.predict(X_test)
cm_dt = confusion_matrix(y_test, y_pred)
print(cm_dt)

dt_result = accuracy_score(y_test,y_pred)
print("Accuracy:",dt_result)

f1_score(y_test,y_pred)
print(classification_report(y_test,y_pred))

recall = cm_dt[0][0]/(cm_dt[0][0] + cm_dt[0][1])
precision = cm_dt[0][0]/(cm_dt[0][0]+cm_dt[1][1])
print("Recall:",recall, "Precision:",precision)

f1_score(y_test,y_pred)
print(classification_report(y_test,y_pred))

#Plot ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
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



