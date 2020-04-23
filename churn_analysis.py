import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import pickle
import sys
from sklearn.metrics import confusion_matrix



if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
    
    
data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(data.head())


for i in range(len(data['TotalCharges'])):
    if data.iloc[i,19] == ' ':
        print(i)
        
data['TotalCharges'] = data['TotalCharges'].replace(' ', np.nan)

data = data[data['TotalCharges'].notnull()]
data = data.reset_index()[data.columns]

data['TotalCharges'] = data['TotalCharges'].astype(float)

replace_columns = [ 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection','TechSupport','StreamingTV', 'StreamingMovies']

for i in replace_columns:
    data[i] = data[i].replace({'No internet service' : 'No'})
    
data['MultipleLines'] = data['MultipleLines'].replace({'No phone service' : 'No'})


print ("Rows     : " ,data.shape[0])
print ("Columns  : " ,data.shape[1])
print ("\nFeatures : \n" ,data.columns.tolist())
print ("\nMissing values :  ", data.isnull().sum().values.sum())
print ("\nUnique values :  \n",data.nunique())


print(data['Contract'].unique())
print(data['PaymentMethod'].unique())
print(data['InternetService'].unique())

slice1 = data.iloc[:,1:8]
slice2 = data.iloc[:,9:15]
slice3 = data.iloc[:,16:17]
slice4 = data.iloc[:,18:]
result = pd.concat([slice1, slice2, slice3, slice4], axis = 1)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

encode_columns = [ 'gender', 'Partner', 'Dependents','PhoneService','MultipleLines','OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Churn']

for i in encode_columns:
    result[i] = le.fit_transform(result[i])

churn = result.iloc[:,-1:]
result = result.iloc[:,:-1]


internet_service = data.iloc[:,8:9]
contract = data.iloc[:,-6:-5]
payment_method = data.iloc[:,-4:-3]
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
internet_service = ohe.fit_transform(internet_service).toarray()
contract = ohe.fit_transform(contract).toarray()
payment_method = ohe.fit_transform(payment_method).toarray()

internet_service = pd.DataFrame(data = internet_service, index = range(len(internet_service)), columns = ['DSL','Fiber optic', 'No internet service'])
contract = pd.DataFrame(data = contract, index = range(len(contract)), columns = ['Month-to-month', 'One year', 'Two year'])
payment_method = pd.DataFrame(data = payment_method, index = range(len(payment_method)), columns = ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check'])

X = pd.concat([result, internet_service, contract, payment_method], axis = 1)
X = X.values
Y = churn.values



from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(X, Y,test_size=0.2, random_state=0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)



#%%

import keras

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout



classifier = Sequential(layers = None, name = None)
print(X_train.shape)

classifier.add(Dense(128, init = 'uniform', activation = 'tanh')) #giris katmani   #input dimm problem
classifier.add(Dense(256, init = 'uniform', activation = 'tanh'))   #gizli katman
classifier.add(Dense(512, init = 'uniform', activation = 'tanh'))
classifier.add(Dense(1024, init = 'uniform', activation = 'tanh'))
classifier.add(Dense(1, init ='uniform', activation = 'sigmoid'))



classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
            
            
classifier.fit(X_train, y_train, epochs = 50)


y_pred = classifier.predict(X_test, use_multiprocessing=True, max_queue_size=1)
y_pred = (y_pred > 0.5)

cm = confusion_matrix(y_test,y_pred)
print('ANN CM\n', cm)


doc = 'model.save'
pickle.dump(classifier, open(doc, 'wb'))

#%% 
