# %%
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %%
file = pd.read_csv('pre_standard.csv')
file 
# %%
file['week']
# %%
type(file['week'])
# %%
type(file)
# %%
from matplotlib import pyplot as plt
sns.set_context('paper', font_scale=2.0)
pp=sns.pairplot(file)
#pp.savefig('sns_pairplot.png')
plt.show()
# %%
cor = file[['date','week','max_tmp','min_tmp','ave_tmp','prec',	'sum_t','snow',	'ave_hmd','ave_cloud','hotcons'	,'coldcons','milk cons']].corr()
sns.heatmap(cor, cmap= sns.color_palette('coolwarm', 10), annot=True, linewidths=0.1, annot_kws={"size":8})
plt.show()
# %%
from sklearn import preprocessing
ss = preprocessing.StandardScaler()
file_standard = ss.fit_transform(file)
type(file_standard)
# %%
X_train=file.iloc[0:171,0:10]
X_test=file.iloc[170:192,0:10]
yh_train=file.loc[0:170,'hotcons']
yh_test=file.loc[170:190,'hotcons']
yc_train=file.loc[0:170,'coldcons']
yc_test=file.loc[170:190,'coldcons']
ya_train=file.loc[0:170,'milk cons']
ya_test=file.loc[170:190,'milk cons']
# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# %%
model = ExtraTreesRegressor(bootstrap=True)
# %% hot prediction
model.fit(X_train, yh_train)
clf=model.fit(X_train, yh_train)

yh_pred_tr = model.predict(X_train)
yh_pred = model.predict(X_test)
# %%
mean_squared_error(yh_test, yh_pred, squared=False)
# %%
r2_score(yh_train, yh_pred_tr)
# %%
def plot_result(y_true_tr, y_pred_tr, y_true_te, y_pred_te, margin=0.1):
    vals = np.concatenate([y_true_tr, y_pred_tr, y_true_te, y_pred_te])
    #https://techacademy.jp/magazine/33340
    l, u = vals.min(), vals.max()
    l = l - margin* (u-l)
    u = u + margin * (u-l)
    
    fig, ax = plt.subplots(1,2, figsize=(10,5))

    ax[0].plot([l, u], [l, u], linestyle='--', c='gray')
    ax[0].plot(y_true_tr, y_pred_tr, 'o', c='blue', alpha=0.5)
    ax[0].set_xlim(l, u)
    ax[0].set_ylim(l, u)
    ax[0].set_aspect('equal')
    ax[0].set_xlabel('y_true')
    ax[0].set_ylabel('y_pred')
    ax[0].set_title('Train')

    ax[1].plot([l, u], [l, u], linestyle='--', c='gray')
    ax[1].plot(y_true_te, y_pred_te, 'o', c='red', alpha=0.5)
    ax[1].set_xlim(l, u)
    ax[1].set_ylim(l, u)
    ax[1].set_aspect('equal')
    ax[1].set_xlabel('y_true')
    ax[1].set_ylabel('y_pred')
    ax[1].set_title('Test')

    plt.tight_layout()
    plt.show()
# %%
plot_result(yh_train, yh_pred_tr, yh_test, yh_pred)
# %% 
df_importance = pd.DataFrame(zip(X_train.columns, clf.feature_importances_),columns=["Features","Importance"])
df_importance = df_importance.sort_values("Importance",ascending=False)

plt.figure(figsize=(10,5))
sns.barplot(x="Importance", y="Features",data=df_importance,ci=None)
plt.title("Gini Importance")
plt.show()
# %%
result = permutation_importance(clf, X_test, yh_test, n_repeats=5)
result
# %%
df_importance = pd.DataFrame(zip(X_train.columns, result["importances"].mean(axis=1)),columns=["Features","Importance"])
df_importance = df_importance.sort_values("Importance",ascending=False)
plt.figure(figsize=(10,5))
sns.barplot(x="Importance", y="Features",data=df_importance,ci=None)
plt.title("Permutation Importance")
plt.show()
# %%
#import eli5
#from eli5.sklearn import PermutationImportance

#perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
#eli5.show_weights(perm, feature_names = val_X.columns.tolist())
# %%
