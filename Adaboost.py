import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc
import datetime
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.metrics import SCORERS
warnings.simplefilter('ignore')
matplotlib.rcParams['figure.dpi'] = 100
sns.set()

# Extract a minibatch from the original training dataset
def TrTeSplit(X_tr, y_tr, size):
    X_trspl, X_tespl, y_trspl, y_tespl = train_test_split(X_tr, y_tr, test_size=size, random_state=0)
    X_trspl = X_trspl.reset_index(drop = True)
    X_tespl = X_tespl.reset_index(drop = True)
    y_trspl = y_trspl.reset_index(drop = True)
    y_tespl = y_tespl.reset_index(drop = True)

    return X_trspl, X_tespl, y_trspl, y_tespl

# Parameter Tunning using cross validation
def model_param_select(X, y, nfolds):
    max_depth = [20, 21, 22]
    min_samples_split = [2, 3]
    min_samples_leaf = [4, 5, 6]
    #the range is being reduced since after several runs
    #I have removed some redundant range value that will yield longer run time
    param_grid = {'max_depth': max_depth, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}
    grid_search = GridSearchCV(DecisionTreeRegressor(random_state = 0), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_

def RMSE(y_test, y_pred):
    rss=((y_test-y_pred)**2).sum()
    mse=np.mean((y_test-y_pred)**2)
    rmse = np.sqrt(np.mean((y_test-y_pred)**2))
    return rmse

def DataPre(new_train):
    X_tr = new_train.drop(columns = ['meter_reading', 'timestamp'])
    y_tr = new_train['meter_reading']
    X_trspl, X_tespl, y_trspl, y_tespl = TrTeSplit(X_tr, y_tr, 0.33)
    del X_tr, y_tr
    return X_trspl, X_tespl, y_trspl, y_tespl


def LearningVisual(model, X_trspl, y_trspl, X_tespl, y_tespl, percentSize):
    # Visualize learning curves
    plt.figure()
    train_sizes, train_scores, test_scores = \
    learning_curve(model, X_trspl, y_trspl, train_sizes=np.linspace(0.1, 1, percentSize),
                   scoring="neg_root_mean_squared_error", cv=10)
    train_sizes, val_train_scores, val_test_scores = \
    learning_curve(model, X_tespl, y_tespl, train_sizes=np.linspace(0.1, 1, percentSize),
                   scoring="neg_root_mean_squared_error", cv=10)
    plt.plot(train_sizes, -test_scores.mean(1), 'o-', color="r",label="train set")
    plt.plot(train_sizes, -val_test_scores.mean(1), 'o-', color ='b', label="validation set")
    plt.xlabel("Dataset size")
    plt.ylabel("Root Mean Squared Error")
    plt.title('Learning curves')
    plt.legend(loc="best")

    plt.show()


# Load Data
new_train = pd.read_csv('sample_train_tiny.csv')
X_trspl, X_tespl, y_trspl, y_tespl = DataPre(new_train)
del new_train
#Implement Tree model with the choosen parameter

model = AdaBoostRegressor(base_estimator = DecisionTreeRegressor(max_depth=20), n_estimators = 50, learning_rate = 1.0)
model.fit(X_trspl, y_trspl)
y_pred = model.predict(X_tespl)

rmse = RMSE(y_tespl, y_pred)
print('rmse for whole dataset: ', rmse)

feat_importances = pd.Series(model.feature_importances_, index=X_trspl.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.title('Feature Important based on Decision Tree regressor')
plt.xlabel('feature scores')
plt.ylabel('feature names')
plt.show()

#Implement Adaboost model with the choosen parameter

model = AdaBoostRegressor(base_estimator = DecisionTreeRegressor(max_depth=20), n_estimators = 50, learning_rate = 1.0)
model.fit(X_trspl, y_trspl)
y_pred = model.predict(X_tespl)


rmse = RMSE(y_tespl, y_pred)
print(rmse)

feat_importances = pd.Series(model.feature_importances_, index=X_trspl.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.title('Feature Important based on AdaBoost regressor')
plt.xlabel('feature scores')
plt.ylabel('feature names')
plt.show()

LearningVisual(model, X_trspl, y_trspl, X_tespl, y_tespl, 10)
