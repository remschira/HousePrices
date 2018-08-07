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
    fileName      = 'train.csv'
    data          = pd.read_csv(path+fileName)    
    features_list = [i for i in data]

    y             = ['SalePrice'] #dependent variable    
    
    data = data.fillna({'Alley':'NoAlley'})#
    data = data.fillna({'MasVnrType':'NoMasonryVeneer'})
    data = data.fillna({'BsmtQual':'NoBasement'})#
    data = data.fillna({'BsmtCond':'NoBasement'})#
    data = data.fillna({'BsmtExposure':'NoBasement'})#
    data = data.fillna({'BsmtFinType1':'NoBasement'})#
    data = data.fillna({'BsmtFinType2':'NoBasement'})#
    data = data.fillna({'Electrical':'None'})
    data = data.fillna({'FireplaceQu':'NoFireplace'})#
    data = data.fillna({'GarageType':'NoGarage'})
    data = data.fillna({'GarageQual':'NoGarage'})#
    data = data.fillna({'GarageCond':'NoGarage'})#
    data = data.fillna({'GarageFinish':'NoGarage'})#    
    data = data.fillna({'PoolQC':'NoPool'})#
    data = data.fillna({'Fence':'NoFence'})#
    data = data.fillna({'MiscFeature':'None'})

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

    data.replace(cleanup_nums, inplace=True)
    data = pd.get_dummies(data, columns=['MSZoning'])
    data = pd.get_dummies(data, columns=['Street'])
    data = pd.get_dummies(data, columns=['LotShape'])
    data = pd.get_dummies(data, columns=['LandContour'])
    data = pd.get_dummies(data, columns=['LotConfig'])
    data = pd.get_dummies(data, columns=['Neighborhood'])
    data = pd.get_dummies(data, columns=['Condition1'])
    data = pd.get_dummies(data, columns=['Condition2'])
    data = pd.get_dummies(data, columns=['BldgType'])
    data = pd.get_dummies(data, columns=['HouseStyle'])
    data = pd.get_dummies(data, columns=['RoofStyle'])
    data = pd.get_dummies(data, columns=['RoofMatl'])
    data = pd.get_dummies(data, columns=['Exterior1st'])
    data = pd.get_dummies(data, columns=['Exterior2nd'])
    data = pd.get_dummies(data, columns=['MasVnrType'])
    data = pd.get_dummies(data, columns=['Foundation'])
    data = pd.get_dummies(data, columns=['Heating'])
    data = pd.get_dummies(data, columns=['Electrical'])
    data = pd.get_dummies(data, columns=['GarageType'])
    data = pd.get_dummies(data, columns=['MiscFeature'])
    data = pd.get_dummies(data, columns=['SaleType'])
    data = pd.get_dummies(data, columns=['SaleCondition'])    

    data['LotFrontage'].fillna(data['LotFrontage'].median(),inplace=True)
    data['MasVnrArea'].fillna(data['MasVnrArea'].median(),inplace=True)
    data['GarageYrBlt'].fillna(data['GarageYrBlt'].median(),inplace=True)        

    X_train,X_test,y_train,y_test = train_test_split(
        data,data[y],test_size=0.33,random_state=rng)

    c,r             = y_train.shape
    y_train_reshape = y_train.values.reshape(c,) 

    #find predictions for X_test with Gradient Boosting
    clf_gbm = GradientBoostingRegressor(n_estimators=350,max_depth=3,learning_rate=0.01,
                                         min_samples_split=4,min_samples_leaf=1)
    clf_gbm.fit(X_train,y_train_reshape)
    pred_gbm = clf_gbm.predict(X_test)
    log_pred_gbm = np.log(pred_gbm)
    log_y_test= np.log(y_test)

    
    importances   = clf_gbm.feature_importances_
    g             = sns.barplot(x=list(X_train), y=importances)

    for item in g.get_xticklabels():
        item.set_rotation(90)







    #print(np.sqrt(mean_squared_error(log_y_test,log_pred_gbm)))


    #data.to_csv('./output/noCategories.csv',index=True)    
    
    #print(data.dtypes)
    #print(data.columns[data.isnull().any()].tolist())
    

    #print(obj_data.head())    
    #print(features_list)
    
    #print(obj_data.columns[obj_data.isnull().any()].tolist())
    
    #print(obj_data[obj_data['Alley'].isnull()])


    
    #data.hist(bins=100)
    #plt.show()
