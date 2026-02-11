from catboost import CatBoostRegressor
import pandas as pd
import numpy as np
#from sklearn.preprocessing import LabelEncoder, StandardScaler
df = pd.read_csv(r"C:\Users\apris\OneDrive\Documents\ML_Kaggle\kaggle_work\road_accident\test\train.csv")
X = df.drop(['accident_risk', 'id'], axis=1)
y = df['accident_risk']
#model= CatBoostRegressor()
categorical_features_indices = np.where(X.dtypes.astype(str).isin(['object','bool']) )[0]
#print(X.dtypes.isin([bool]))
#print(X.dtypes.isin([object]))
#print(X.dtypes.astype(str).isin(['object','bool']))
for i in categorical_features_indices:
    print(X.columns[i])
model = CatBoostRegressor(cat_features=categorical_features_indices)
model.fit(X, y)
X_test = pd.read_csv(r"C:\Users\apris\OneDrive\Documents\ML_Kaggle\kaggle_work\road_accident\test\test (2).csv")
y_test = model.predict(X_test.drop(['id'], axis=1))
output = pd.DataFrame({'id': X_test['id'], 'accident_risk': y_test})
output.to_csv('catboost_model.csv', index=False)