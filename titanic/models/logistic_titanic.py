import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import pandas as pd
clf = LogisticRegression(max_iter=10000000,random_state=0)
df = pd.read_csv("train.csv")
label = LabelEncoder()
df['Sex'] = label.fit_transform(df['Sex'])
print(df['Sex'])
onehot = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
embark= onehot.fit_transform(df['Embarked'].values.reshape(-1,1))
embark_df= pd.DataFrame(embark,columns=onehot.get_feature_names_out(['Embarked']))
df = pd.concat([df.drop('Embarked',axis=1),embark_df],axis=1)
Y_train = df['Survived']
X_train = df.drop(['Survived','Name','Ticket','Cabin'],axis=1)
X_train.drop(['Embarked_nan'],axis=1,inplace=True)
for i in X_train.columns:
    X_train[i].fillna(X_train[i].mean(),inplace=True)
clf.fit(X_train, Y_train)
df = pd.read_csv("test.csv")
label = LabelEncoder()
df['Sex'] = label.fit_transform(df['Sex'])
print(df['Sex'])
embark= onehot.fit_transform(df['Embarked'].values.reshape(-1,1))
embark_df= pd.DataFrame(embark,columns=onehot.get_feature_names_out(['Embarked']))
df = pd.concat([df.drop('Embarked',axis=1),embark_df],axis=1)
X_test = df.drop(['Name','Ticket','Cabin'],axis=1)
for i in X_test.columns:
    X_test[i].fillna(X_test[i].mean(),inplace=True)
output=clf.predict(X_test)
df1=pd.concat([df['PassengerId'],pd.Series(output,name='Survived')],axis=1)
df1.to_csv('logistic_titanic.csv',index=False)
print(df1)
#print(clf.predict(X_test))