import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC,SVC
from nameparser import HumanName
from sklearn import preprocessing

if __name__ == '__main__':

    rng           = np.random.RandomState(42)

    #read data
    #we will split the train.csv data into training and testing data
    path          = './data/'
    fileName      = 'train.csv'
    data          = pd.read_csv(path+fileName)    
    features_list = [i for i in data]

    obj_data = data.select_dtypes(include=['object']).copy()
    #print(obj_data.head())
    features_list = [i for i in obj_data]
    
    obj_data = obj_data.fillna({'Alley':'NoAlley'})#
    obj_data = obj_data.fillna({'MasVnrType':'NoMasonryVeneer'})
    obj_data = obj_data.fillna({'BsmtQual':'NoBasement'})#
    obj_data = obj_data.fillna({'BsmtCond':'NoBasement'})#
    obj_data = obj_data.fillna({'BsmtExposure':'NoBasement'})#
    obj_data = obj_data.fillna({'BsmtFinType1':'NoBasement'})#
    obj_data = obj_data.fillna({'BsmtFinType2':'NoBasement'})#
    obj_data = obj_data.fillna({'Electrical':'None'})
    obj_data = obj_data.fillna({'FireplaceQu':'NoFireplace'})#
    obj_data = obj_data.fillna({'GarageType':'NoGarage'})
    obj_data = obj_data.fillna({'GarageQual':'NoGarage'})#
    obj_data = obj_data.fillna({'GarageCond':'NoGarage'})#
    obj_data = obj_data.fillna({'GarageFinish':'NoGarage'})#    
    obj_data = obj_data.fillna({'PoolQC':'NoPool'})#
    obj_data = obj_data.fillna({'Fence':'NoFence'})#
    obj_data = obj_data.fillna({'MiscFeature':'None'})

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
                                  'ELO': 1},
                    'LandSlope': {'Gtl': 3, 'Mod': 3, 'Sev': 1},
                    'ExterQual': {'Ex': 5, 'Gd': 4, 'TA': 3,
                                  'Fa': 2, 'Po': 1},
                    'ExterCond': {'Ex': 5, 'Gd': 4, 'TA': 3,
                                  'Fa': 2, 'Po': 1},
                    'HeatingQC': {'Ex': 5, 'Gd': 4, 'TA': 3,
                                  'Fa': 2, 'Po': 1},
                    'CentralAir': {'Y': 1, 'N': 0},
                    'KitchenQual': {'Ex': 5, 'Gd': 4, 'TA': 3,
                                    'Fa': 2, 'Po': 1},
                    'Functional': {'Typ': 8, 'Min1': 7, 'Min2': 6,
                                   'Mod': 5, 'Maj1': 4, 'Maj2': 3,
                                   'Sev': 2, 'Sal': 1},
                    'PavedDrive': {'Y': 2, 'P': 1, 'N': 0}
                    
                    
                    
                    
                    }

                       
    obj_data.replace(cleanup_nums, inplace=True)
    obj_data = pd.get_dummies(obj_data, columns=['MSZoning'])
    obj_data = pd.get_dummies(obj_data, columns=['Street'])
    obj_data = pd.get_dummies(obj_data, columns=['LotShape'])
    obj_data = pd.get_dummies(obj_data, columns=['LandContour'])
    obj_data = pd.get_dummies(obj_data, columns=['LotConfig'])
    obj_data = pd.get_dummies(obj_data, columns=['Neighborhood'])
    obj_data = pd.get_dummies(obj_data, columns=['Condition1'])
    obj_data = pd.get_dummies(obj_data, columns=['Condition2'])
    obj_data = pd.get_dummies(obj_data, columns=['BldgType'])
    obj_data = pd.get_dummies(obj_data, columns=['HouseStyle'])
    obj_data = pd.get_dummies(obj_data, columns=['RoofStyle'])
    obj_data = pd.get_dummies(obj_data, columns=['RoofMatl'])
    obj_data = pd.get_dummies(obj_data, columns=['Exterior1st'])
    obj_data = pd.get_dummies(obj_data, columns=['Exterior2nd'])
    obj_data = pd.get_dummies(obj_data, columns=['MasVnrType'])
    obj_data = pd.get_dummies(obj_data, columns=['Foundation'])
    obj_data = pd.get_dummies(obj_data, columns=['Heating'])
    obj_data = pd.get_dummies(obj_data, columns=['Electrical'])
    obj_data = pd.get_dummies(obj_data, columns=['GarageType'])
    obj_data = pd.get_dummies(obj_data, columns=['MiscFeature'])
    obj_data = pd.get_dummies(obj_data, columns=['SaleType'])
    obj_data = pd.get_dummies(obj_data, columns=['SaleCondition'])    
    
    


    
    
    #print(obj_data.head())
    print(obj_data.dtypes)


    #print(features_list)
    
    #print(obj_data.columns[obj_data.isnull().any()].tolist())
    
    #print(obj_data[obj_data['Alley'].isnull()])


    
    #data.hist(bins=100)
    #plt.show()


################

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
    data    = pd.read_csv(path+train_file)    
    test_data     = pd.read_csv(path+test_file)        
    train_data    = pd.read_csv(path+train_file)
    test_data     = pd.read_csv(path+test_file)        
    features_list = [i for i in train_data]


    y             = ['SalePrice'] #dependent variable    
    
    train_data = train_data.fillna({'Alley':'NoAlley'})#
    train_data = train_data.fillna({'MasVnrType':'NoMasonryVeneer'})
    train_data = train_data.fillna({'BsmtQual':'NoBasement'})#
    train_data = train_data.fillna({'BsmtCond':'NoBasement'})#
    train_data = train_data.fillna({'BsmtExposure':'NoBasement'})#
    train_data = train_data.fillna({'BsmtFinType1':'NoBasement'})#
    train_data = train_data.fillna({'BsmtFinType2':'NoBasement'})#
    train_data = train_data.fillna({'Electrical':'None'})
    train_data = train_data.fillna({'FireplaceQu':'NoFireplace'})#
    train_data = train_data.fillna({'GarageType':'NoGarage'})
    train_data = train_data.fillna({'GarageQual':'NoGarage'})#
    train_data = train_data.fillna({'GarageCond':'NoGarage'})#
    train_data = train_data.fillna({'GarageFinish':'NoGarage'})#    
    train_data = train_data.fillna({'PoolQC':'NoPool'})#
    train_data = train_data.fillna({'Fence':'NoFence'})#
    train_data = train_data.fillna({'MiscFeature':'None'})

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
                                  'ELO': 1},
                    'LandSlope': {'Gtl': 3, 'Mod': 3, 'Sev': 1},
                    'ExterQual': {'Ex': 5, 'Gd': 4, 'TA': 3,
                                  'Fa': 2, 'Po': 1},
                    'ExterCond': {'Ex': 5, 'Gd': 4, 'TA': 3,
                                  'Fa': 2, 'Po': 1},
                    'HeatingQC': {'Ex': 5, 'Gd': 4, 'TA': 3,
                                  'Fa': 2, 'Po': 1},
                    'CentralAir': {'Y': 1, 'N': 0},
                    'KitchenQual': {'Ex': 5, 'Gd': 4, 'TA': 3,
                                    'Fa': 2, 'Po': 1},
                    'Functional': {'Typ': 8, 'Min1': 7, 'Min2': 6,
                                   'Mod': 5, 'Maj1': 4, 'Maj2': 3,
                                   'Sev': 2, 'Sal': 1},
                    'PavedDrive': {'Y': 2, 'P': 1, 'N': 0}
                    }

    train_data.replace(cleanup_nums, inplace=True)
    train_data = pd.get_dummies(train_data, columns=['MSZoning'])
    train_data = pd.get_dummies(train_data, columns=['Street'])
    train_data = pd.get_dummies(train_data, columns=['LotShape'])
    train_data = pd.get_dummies(train_data, columns=['LandContour'])
    train_data = pd.get_dummies(train_data, columns=['LotConfig'])
    train_data = pd.get_dummies(train_data, columns=['Neighborhood'])
    train_data = pd.get_dummies(train_data, columns=['Condition1'])
    train_data = pd.get_dummies(train_data, columns=['Condition2'])
    train_data = pd.get_dummies(train_data, columns=['BldgType'])
    train_data = pd.get_dummies(train_data, columns=['HouseStyle'])
    train_data = pd.get_dummies(train_data, columns=['RoofStyle'])
    train_data = pd.get_dummies(train_data, columns=['RoofMatl'])
    train_data = pd.get_dummies(train_data, columns=['Exterior1st'])
    train_data = pd.get_dummies(train_data, columns=['Exterior2nd'])
    train_data = pd.get_dummies(train_data, columns=['MasVnrType'])
    train_data = pd.get_dummies(train_data, columns=['Foundation'])
    train_data = pd.get_dummies(train_data, columns=['Heating'])
    train_data = pd.get_dummies(train_data, columns=['Electrical'])
    train_data = pd.get_dummies(train_data, columns=['GarageType'])
    train_data = pd.get_dummies(train_data, columns=['MiscFeature'])
    train_data = pd.get_dummies(train_data, columns=['SaleType'])
    train_data = pd.get_dummies(train_data, columns=['SaleCondition'])    


    train_data['LotFrontage'].fillna(train_data['LotFrontage'].median(),inplace=True)
    train_data['MasVnrArea'].fillna(train_data['MasVnrArea'].median(),inplace=True)
    train_data['GarageYrBlt'].fillna(train_data['GarageYrBlt'].median(),inplace=True)        


    test_data = test_data.fillna({'Alley':'NoAlley'})#
    test_data = test_data.fillna({'MasVnrType':'NoMasonryVeneer'})
    test_data = test_data.fillna({'BsmtQual':'NoBasement'})#
    test_data = test_data.fillna({'KitchenQual':'NoKitchen'})#    
    test_data = test_data.fillna({'BsmtCond':'NoBasement'})#
    test_data = test_data.fillna({'BsmtExposure':'NoBasement'})#
    test_data = test_data.fillna({'BsmtFinType1':'NoBasement'})#
    test_data = test_data.fillna({'BsmtFinType2':'NoBasement'})#
    test_data = test_data.fillna({'Electrical':'None'})
    test_data = test_data.fillna({'FireplaceQu':'NoFireplace'})#
    test_data = test_data.fillna({'GarageType':'NoGarage'})
    test_data = test_data.fillna({'GarageQual':'NoGarage'})#
    test_data = test_data.fillna({'GarageCond':'NoGarage'})#
    test_data = test_data.fillna({'GarageFinish':'NoGarage'})#    
    test_data = test_data.fillna({'PoolQC':'NoPool'})#
    test_data = test_data.fillna({'Fence':'NoFence'})#
    test_data = test_data.fillna({'MiscFeature':'None'})
    test_data = test_data.fillna({'Utilities':'None'})    
    
    test_data.replace(cleanup_nums, inplace=True)
    test_data = pd.get_dummies(data, columns=['MSZoning'])
    test_data = pd.get_dummies(data, columns=['Street'])
    test_data = pd.get_dummies(data, columns=['LotShape'])
    test_data = pd.get_dummies(data, columns=['LandContour'])
    test_data = pd.get_dummies(data, columns=['LotConfig'])
    test_data = pd.get_dummies(data, columns=['Neighborhood'])
    test_data = pd.get_dummies(data, columns=['Condition1'])
    test_data = pd.get_dummies(data, columns=['Condition2'])
    test_data = pd.get_dummies(data, columns=['BldgType'])
    test_data = pd.get_dummies(data, columns=['HouseStyle'])
    test_data = pd.get_dummies(data, columns=['RoofStyle'])
    test_data = pd.get_dummies(data, columns=['RoofMatl'])
    test_data = pd.get_dummies(data, columns=['Exterior1st'])
    test_data = pd.get_dummies(data, columns=['Exterior2nd'])
    test_data = pd.get_dummies(data, columns=['MasVnrType'])
    test_data = pd.get_dummies(data, columns=['Foundation'])
    test_data = pd.get_dummies(data, columns=['Heating'])
    test_data = pd.get_dummies(data, columns=['Electrical'])
    test_data = pd.get_dummies(data, columns=['GarageType'])
    test_data = pd.get_dummies(data, columns=['MiscFeature'])
    test_data = pd.get_dummies(data, columns=['SaleType'])
    test_data = pd.get_dummies(data, columns=['SaleCondition'])    



    
    test_data['LotFrontage'].fillna(test_data['LotFrontage'].median(),inplace=True)
    test_data['MasVnrArea'].fillna(test_data['MasVnrArea'].median(),inplace=True)
    test_data['GarageYrBlt'].fillna(test_data['GarageYrBlt'].median(),inplace=True)
    test_data['BsmtFinSF1'].fillna(test_data['BsmtFinSF1'].median(),inplace=True)
    test_data['BsmtFinSF2'].fillna(test_data['BsmtFinSF2'].median(),inplace=True)
    test_data['BsmtUnfSF'].fillna(test_data['BsmtUnfSF'].median(),inplace=True)
    test_data['TotalBsmtSF'].fillna(test_data['TotalBsmtSF'].median(),inplace=True)
    test_data['BsmtFullBath'].fillna(test_data['BsmtFullBath'].mode()[0],inplace=True)
    test_data['BsmtHalfBath'].fillna(test_data['BsmtHalfBath'].mode()[0],inplace=True)
    test_data['Functional'].fillna(test_data['Functional'].mode()[0],inplace=True)
    test_data['GarageCars'].fillna(test_data['GarageCars'].mode()[0],inplace=True)
    test_data['GarageArea'].fillna(test_data['GarageArea'].median(),inplace=True)    
    X_test  = test_data

    print(test_data.columns[test_data.isnull().any()].tolist())    
    print(test_data.shape)
    print(train_data.shape)    
    exit()

    
    X_train,X_blank,y_train,y_blank = train_test_split(
        train_data,train_data[y],test_size=0.33,random_state=rng)

    c,r             = y_train.shape
    y_train_reshape = y_train.values.reshape(c,) 

    #find predictions for X_test with Gradient Boosting
    clf_gbm = GradientBoostingRegressor(n_estimators=350,max_depth=3,learning_rate=0.01,
                                         min_samples_split=4,min_samples_leaf=1)
    clf_gbm.fit(X_train,y_train_reshape)
    pred_gbm = clf_gbm.predict(X_test)
    log_pred_gbm = np.log(pred_gbm)
#    log_y_test= np.log(y_test)

    

    #data.to_csv('./output/noCategories.csv',index=True)    
    
    #print(data.dtypes)
    #print(data.columns[data.isnull().any()].tolist())
    

    #print(obj_data.head())    
    #print(features_list)
    
    #print(obj_data.columns[obj_data.isnull().any()].tolist())
    
    #print(obj_data[obj_data['Alley'].isnull()])


    
    #data.hist(bins=100)
    #plt.show()

