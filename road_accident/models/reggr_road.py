import pandas as pd 
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.svm import LinearSVR,SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
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
model= RandomForestRegressor(n_estimators=100)
model.fit(X,Y)  

#training
df = pd.read_csv("road_accident/test/test(2).csv")
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