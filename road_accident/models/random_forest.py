import pandas as pd 
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.svm import LinearSVR,SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
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
# model = SGDRegressor(loss='squared_error', max_iter=10000)
# model.fit(X, Y)
# print(model.predict(X))
# print(mean_squared_error(Y,model.predict(X)))  note: sgd gave me 0.007 as mse

#note: rf gave me 0.0005 as mse which is better which proves that the data is non linear in nature and hence rf is better suited since it branches on diff thresholds at each level
#rd not that much affected by multicollinearity since it splits and doesnt learn weights like sgd or svr

 #tried shap but too slow and moreover it only tells me which feature is imp etc, not the hyperparameter
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 150)
    max_depth = trial.suggest_int('max_depth', 5, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )

    rf.fit(X, Y)
    preds = rf.predict(X)
    mse = mean_squared_error(Y, preds)
    return mse

study=optuna.create_study(direction='minimize',study_name='rf')
study.optimize(func=objective,n_trials=1)
print(study.best_params)
print(study.best_value)
param=study.best_params
model= RandomForestRegressor(**param)
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
output.to_csv('road_accident/outputs/submission_rf.csv',index=False)