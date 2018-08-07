import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score,mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC,SVC
from nameparser import HumanName
from sklearn import preprocessing

if __name__ == '__main__':

    rng           = np.random.RandomState(42)

    #read data
    #we will split the train.csv data into training and testing data
    path          = './data/'
    train_file    = 'train.csv'
    test_file     = 'test.csv'    
    train_data    = pd.read_csv(path+train_file)
    test_data     = pd.read_csv(path+test_file)
    train_features= train_data.drop(['SalePrice'],axis=1)
    
    features          = train_features.append(test_data)

       
    features_list = [i for i in train_data]


    y             = ['SalePrice'] #dependent variable    
    
    features = features.fillna({'Alley':'NoAlley'})#
    features = features.fillna({'MasVnrType':'NoMasonryVeneer'})
    features = features.fillna({'BsmtQual':'NoBasement'})#
    features = features.fillna({'BsmtCond':'NoBasement'})#
    features = features.fillna({'BsmtExposure':'NoBasement'})#
    features = features.fillna({'BsmtFinType1':'NoBasement'})#
    features = features.fillna({'BsmtFinType2':'NoBasement'})#
    features = features.fillna({'Electrical':'None'})
    features = features.fillna({'FireplaceQu':'NoFireplace'})#
    features = features.fillna({'GarageType':'NoGarage'})
    features = features.fillna({'GarageQual':'NoGarage'})#
    features = features.fillna({'GarageCond':'NoGarage'})#
    features = features.fillna({'GarageFinish':'NoGarage'})#    
    features = features.fillna({'PoolQC':'NoPool'})#
    features = features.fillna({'Fence':'NoFence'})#
    features = features.fillna({'MiscFeature':'None'})
    features = features.fillna({'Utilities':'None'})
    features = features.fillna({'KitchenQual':'None'})        

    cleanup_nums = {'Alley': {'Grvl': 2, 'Pave': 1, 'NoAlley': 0},
                    'BsmtQual': {'Ex': 5, 'Gd': 4, 'TA': 3,
                                 'Fa': 2, 'Po': 1, 'NoBasement': 0},
                    'BsmtCond': {'Ex': 5, 'Gd': 4, 'TA': 3,
                                 'Fa': 2, 'Po': 1, 'NoBasement': 0},
                    'BsmtExposure': {'Gd': 4, 'Av': 3, 'Mn': 2,
                                     'No': 1, 'NoBasement': 0},
                    'BsmtFinType1': {'GLQ': 6, 'ALQ': 5, 'BLQ': 4,
                                     'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NoBasement': 0},
                    'BsmtFinType2': {'GLQ': 6, 'ALQ': 5, 'BLQ': 4,
                                     'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NoBasement': 0},
                    'FireplaceQu': {'Ex': 5, 'Gd': 4, 'TA': 3,
                                    'Fa': 2, 'Po': 1, 'NoFireplace': 0},
                    'GarageQual': {'Ex': 5, 'Gd': 4, 'TA': 3,
                                 'Fa': 2, 'Po': 1, 'NoGarage': 0},
                    'GarageCond': {'Ex': 5, 'Gd': 4, 'TA': 3,
                                   'Fa': 2, 'Po': 1, 'NoGarage': 0},
                    'GarageFinish': {'Fin': 3, 'RFn': 2, 'Unf': 1, 'NoGarage': 0},
                    'PoolQC': {'Ex': 4, 'Gd': 3, 'TA': 2,
                                 'Fa': 1, 'NoPool': 0},
                    'Fence': {'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2,
                                 'MnWw': 1, 'NoFence': 0},
                    'Utilities': {'AllPub': 4, 'NoSewr': 3, 'NoSeWa': 2,
                                  'ELO': 1, 'None': 0},
                    'LandSlope': {'Gtl': 3, 'Mod': 3, 'Sev': 1},
                    'ExterQual': {'Ex': 5, 'Gd': 4, 'TA': 3,
                                  'Fa': 2, 'Po': 1},
                    'ExterCond': {'Ex': 5, 'Gd': 4, 'TA': 3,
                                  'Fa': 2, 'Po': 1},
                    'HeatingQC': {'Ex': 5, 'Gd': 4, 'TA': 3,
                                  'Fa': 2, 'Po': 1},
                    'CentralAir': {'Y': 1, 'N': 0},
                    'KitchenQual': {'Ex': 5, 'Gd': 4, 'TA': 3,
                                    'Fa': 2, 'Po': 1, 'None': 0},
                    'Functional': {'Typ': 8, 'Min1': 7, 'Min2': 6,
                                   'Mod': 5, 'Maj1': 4, 'Maj2': 3,
                                   'Sev': 2, 'Sal': 1},
                    'PavedDrive': {'Y': 2, 'P': 1, 'N': 0}
                    }

    features.replace(cleanup_nums, inplace=True)
    features = pd.get_dummies(features, columns=['MSZoning'])
    features = pd.get_dummies(features, columns=['Street'])
    features = pd.get_dummies(features, columns=['LotShape'])
    features = pd.get_dummies(features, columns=['LandContour'])
    features = pd.get_dummies(features, columns=['LotConfig'])
    features = pd.get_dummies(features, columns=['Neighborhood'])
    features = pd.get_dummies(features, columns=['Condition1'])
    features = pd.get_dummies(features, columns=['Condition2'])
    features = pd.get_dummies(features, columns=['BldgType'])
    features = pd.get_dummies(features, columns=['HouseStyle'])
    features = pd.get_dummies(features, columns=['RoofStyle'])
    features = pd.get_dummies(features, columns=['RoofMatl'])
    features = pd.get_dummies(features, columns=['Exterior1st'])
    features = pd.get_dummies(features, columns=['Exterior2nd'])
    features = pd.get_dummies(features, columns=['MasVnrType'])
    features = pd.get_dummies(features, columns=['Foundation'])
    features = pd.get_dummies(features, columns=['Heating'])
    features = pd.get_dummies(features, columns=['Electrical'])
    features = pd.get_dummies(features, columns=['GarageType'])
    features = pd.get_dummies(features, columns=['MiscFeature'])
    features = pd.get_dummies(features, columns=['SaleType'])
    features = pd.get_dummies(features, columns=['SaleCondition'])    


    features['LotFrontage'].fillna(features['LotFrontage'].median(),inplace=True)
    features['MasVnrArea'].fillna(features['MasVnrArea'].median(),inplace=True)
    features['GarageYrBlt'].fillna(features['GarageYrBlt'].median(),inplace=True)        
    features['BsmtFinSF1'].fillna(features['BsmtFinSF1'].median(),inplace=True)
    features['BsmtFinSF2'].fillna(features['BsmtFinSF2'].median(),inplace=True)
    features['BsmtUnfSF'].fillna(features['BsmtUnfSF'].median(),inplace=True)
    features['TotalBsmtSF'].fillna(features['TotalBsmtSF'].median(),inplace=True)
    features['BsmtFullBath'].fillna(features['BsmtFullBath'].mode()[0],inplace=True)
    features['BsmtHalfBath'].fillna(features['BsmtHalfBath'].mode()[0],inplace=True)
    features['Functional'].fillna(features['Functional'].mode()[0],inplace=True)
    features['GarageCars'].fillna(features['GarageCars'].mode()[0],inplace=True)
    features['GarageArea'].fillna(features['GarageArea'].median(),inplace=True)    

    features.to_csv('./output/features.csv',index=True)

    train_features = features[features['Id'] <= 1460]
    test_features  = features[features['Id'] > 1460]    

    
    X_train,X_blank,y_train,y_blank = train_test_split(
        train_features,train_data[y],test_size=0.33,random_state=rng)
    X_test = test_features

    
    c,r             = y_train.shape
    y_train_reshape = y_train.values.reshape(c,) 


    clf_gbm = GradientBoostingRegressor(n_estimators=350,max_depth=3,learning_rate=0.01,
                                         min_samples_split=4,min_samples_leaf=1)
    clf_gbm.fit(X_train,y_train_reshape)
    pred_gbm = clf_gbm.predict(X_test)

    pred = pd.DataFrame({'Id':test_features['Id'],'SalePrice':pred_gbm})

    pred.to_csv('./output/prices_submission.csv',index=False)


    


