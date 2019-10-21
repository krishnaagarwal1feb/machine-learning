#%%import libraries 
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

#%% Importing the dataset
train_dataset = pd.read_csv('train.csv')
X_train = train_dataset.iloc[:, :-1]
y_train = train_dataset.loc[:,'SalePrice']
X_test = pd.read_csv('test.csv')

"""if we add .values attribute to the end of the X_train then it would 
convert it to a numpy array"""
#%%feature selection and feature engineering 
features_list = train_dataset.keys()
#Fireplace Qu should be removed or not ?
remove = ['Alley','Utilities','MiscFeature','Fence','PoolQC','PoolArea','FireplaceQu']
X_train.drop(remove,axis=1,inplace=True)
X_train.set_index('Id',inplace=True)

X_test.drop(remove,axis=1,inplace=True)
X_test.set_index('Id',inplace=True)

#%% to count column wise the number of data points to be NaN
cnt =X_train.isna().sum()
cnttest = X_test.isna().sum()
#%%to check which columns have categorical data 
cols = X_train.columns
num_ft = X_train._get_numeric_data().columns
cat_ft = list(set(cols) - set(num_ft))

#%% Taking care of missing data
"""from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 1)
imputer = imputer.fit(X_train)
X_train = imputer.transform(X_train)
"""
#%% Here i can seperate both categorical and numerical data in two different dataframes

cat_df = pd.DataFrame()
for ft in cat_ft:    
    cat_df[ft] = X_train.loc[:,ft]

num_df = pd.DataFrame()
for ft in num_ft:
    num_df[ft] = X_train.loc[:,ft]

cat_df_test = pd.DataFrame()
for ft in cat_ft:    
    cat_df_test[ft] = X_test.loc[:,ft]

num_df_test = pd.DataFrame()
for ft in num_ft:
    num_df_test[ft] = X_test.loc[:,ft]

#%%    task1
    #task1 - encode cat data
    #task2 - impute num data
    #task3 - fill na in cat data and then encode it 
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 1)
imputer = imputer.fit(num_df)
num_df = imputer.transform(num_df)

num_df_test = imputer.transform(num_df_test)

#%%scale down the values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(num_df)
num_df = sc.transform(num_df)
num_df_test = sc.transform(num_df_test)
#now we have to transform the data to normalize the values
#%%
#cat_df.drop('FireplaceQu',axis=1,inplace=True)
cat_df_without_na = cat_df.fillna(method='ffill')
miss_cat = cat_df.isna().sum()
cat_df_test_without_na = cat_df_test.fillna(method='ffill')

#%%    task2
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
encoder = LabelEncoder()
cat_ft_new = cat_df_without_na.keys()
for ft in cat_ft_new:
    cat_df_without_na.loc[:,ft] = encoder.fit_transform(cat_df_without_na.loc[:,ft])
    cat_df_test_without_na.loc[:,ft] = encoder.fit_transform(cat_df_test_without_na.loc[:,ft])

#catdf without na     and num_df
#%%one hot encode the cat data
li = [x for x in range(len(cat_ft)-1)]
onehotencoder = OneHotEncoder(categorical_features = li) 
cat_df_without_na_encoded = onehotencoder.fit_transform(cat_df_without_na).toarray() 
cat_df_test_without_na_encoded = onehotencoder.transform(cat_df_test_without_na).toarray() 
#%%final numpy array
data = np.concatenate((cat_df_without_na_encoded , num_df),axis=1)
test_data = np.concatenate((cat_df_test_without_na_encoded, num_df_test),axis=1)
df_data = pd.DataFrame(data)

#final training data
#%% FEATURE ENGINEERING  
#we can make a new feature which will check how many years old the house is



#%%
from sklearn.feature_selection import SelectKBest
skb = SelectKBest(k=130)
topft = skb.fit_transform(df_data, y_train)
data = topft
topft_test = skb.transform(test_data)
test_data = topft_test
#%%plot data to check feature importance 
plotdf = X_train.dropna()

#%% train test split
"""
from sklearn.model_selection import train_test_split
X_training, X_testing, y_training, y_testing = train_test_split(data, y_train, test_size=0.33, random_state=42)
"""
#%%now we can apply more modelsS
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor()
rf_reg.fit(data, y_train)
y_pred = rf_reg.predict(test_data)
li = [x+1461 for x in range(len(y_pred))]
res = pd.DataFrame(y_pred)
res['id'] = li
res.set_index('id',inplace=True)
res.to_csv('res.csv')
"""#%%
from sklearn.metrics import accuracy_score, recall_score, precision_score
acc = accuracy_score(y_testing, y_pred)
prec = precision_score(y_testing, y_pred)
rec = recall_score(y_testing, y_pred)
"""
