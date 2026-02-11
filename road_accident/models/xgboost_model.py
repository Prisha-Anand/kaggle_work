import pandas as pd 
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.svm import LinearSVR,SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import shap
import optuna
df = pd.read_csv("road_accident/test/train.csv")
#print(df.dtypes)
label=LabelEncoder()
for i in df.dtypes.keys() :
    if(df.dtypes[i]=='object' or df.dtypes[i]=='bool'):
        df[i]=label.fit_transform(df[i])
print(df.dtypes)
Y=df['accident_risk']
X=df.drop(['accident_risk','id'],axis=1)   
X=StandardScaler().fit_transform(X)
 #Trying XGBOOST since its better than your trad gradient boost which operates sequentially whereas here pruning is done and diff features can be handled
 #to identify best split using similarity scores and regularization is also there to avoid overfitting additionally helps me for categ+numer data
 #catboost/adaboost might also work
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 500)
    max_depth = trial.suggest_int('max_depth', 4, 10)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.07)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    tree_method = trial.suggest_categorical('tree_method', ['auto', 'exact', 'approx', 'hist'])
    alpha = trial.suggest_float('alpha', 0.0, 1.0)
    lambda_ = trial.suggest_float('lambda', 0.0, 1.0)
    gamma = trial.suggest_float('gamma', 0.0, 1.0)
    #eta = trial.suggest_float('eta', 0.01, 0.3)
    xgb = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        tree_method=tree_method,
        random_state=42,
        alpha=alpha,
        reg_lambda=lambda_,
        gamma=gamma
    )
    X_train,X_val,Y_train,Y_val=train_test_split(X,Y,test_size=0.2,random_state=42)
    xgb.fit(X_train, Y_train)
    preds = xgb.predict(X_val)
    mse = mean_squared_error(Y_val, preds)
    return mse
study = optuna.create_study(direction="minimize",study_name="xgb")
study.optimize(func=objective,n_trials=20)
print(study.best_params)
print(study.best_value)
model= XGBRegressor(**study.best_params)
model.fit(X,Y)  

#testing
df = pd.read_csv("road_accident/test/test (2).csv")
#print(df.dtypes)
label=LabelEncoder()
for i in df.dtypes.keys() :
    if(df.dtypes[i]=='object' or df.dtypes[i]=='bool'):
        df[i]=label.fit_transform(df[i])
id=df['id']
X=df.drop(['id'],axis=1)   
X=StandardScaler().fit_transform(X)
pred=model.predict(X)

output=pd.DataFrame({'id':id,'accident_risk':pred})
output.to_csv('road_accident/outputs/submission_xgb.csv',index=False)
#gives me 0.05559 on leaderboard which is better than previous models