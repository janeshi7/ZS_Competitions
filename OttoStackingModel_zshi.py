#Author: Zhiyin Shi @University of Chicago

#Otto, a Kaggle Competition

#Problem description: For this competition, we have provided a dataset with 93 features for more than 200,000 
#products. The objective is to build a predictive model which is able to distinguish between our main product 
#categories. The winning models will be open sourced.


# coding: utf-8

# In[10]:

import pandas as pd
import numpy as np

#error measure, cv, train/test split
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from sklearn import model_selection as ms

#models
import xgboost
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#Packages used for selecting best num of clusters for engineered kmean features
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, pdist
from sklearn.preprocessing import scale


# In[3]:

#Import dat
ottoTrain = pd.read_csv("/Users/JaneShi/Desktop/MSCA31009/Project/supervized_classification_data/train_sample.csv")
ottoTest = pd.read_csv("/Users/JaneShi/Desktop/MSCA31009/Project/supervized_classification_data/test_sample.csv")


# In[4]:

#Split id and features in test dat
ottoTestId = ottoTest.iloc[:, 0]
ottoTestDat= ottoTest.iloc[:, 1:]
#Split features and responese in train dat
ottoX = ottoTrain.iloc[:, :93]
ottoY = ottoTrain.iloc[:, 93]
ottoYDummy = pd.get_dummies(ottoY)


# In[5]:

#Engineered Feature 1: count number of non-zeros per row
ottoTrainF1 = (ottoX != 0).astype(int).sum(axis=1)
ottoTrainF1 = ottoTrainF1.rename("newFeature1")
ottoTestF1  = (ottoTestDat != 0).astype(int).sum(axis=1)
ottoTestF1  = ottoTestF1.rename("newFeature1")


# In[6]:

#Engineered Feature 2: if a feature is zero or not
ottoTrainF2 = (ottoX != 0).astype(int)
ottoTrainF2 = ottoTrainF2.add_suffix("_f2")
ottoTestF2  = (ottoTestDat != 0).astype(int)
ottoTestF2 = ottoTestF2.add_suffix("_f2")


# In[ ]:

#Engineered Feature 3: Choose best num of clusters
kmeans_model = [KMeans(n_clusters=k).fit(ottoX) for k in range(9, 52, 3)]
centroids = [c.cluster_centers_ for c in kmeans_model]

k_euclid = [cdist(ottoX, i, "euclidean") for i in centroids]
dist = [np.min(ki, axis = 1) for ki in k_euclid]

#Total within cluster sum of squares
wcss = [sum(d **2) for d in dist]
#Total sum of squares
tss = sum(pdist(ottoX)**2) / ottoX.shape[0]
#Total between-cluster sum of squares
bss = tss - wcss

var_explain = bss/tss
print(var_explain)

plt.figure()
plt.plot(var_explain)
plt.show()
#Best cluster = 30


# In[7]:

#Engineered Feature 3: assign each observation into one of the 30 clusters
kmFinal = KMeans(n_clusters=30)
kmFinal.fit(ottoX)
ottoTrainF3 = pd.Series(kmFinal.predict(ottoX))
ottoTrainF3 = ottoTrainF3.rename("newFeature3")
ottoTestF3  = pd.Series(kmFinal.predict(ottoTestDat))
ottoTestF3 = ottoTestF3.rename("newFeature3")

ottoTrainF3 = pd.get_dummies(ottoTrainF3)
ottoTestF3 = pd.get_dummies(ottoTestF3)
ottoTrainF3 = ottoTrainF3.add_suffix("_f3")
ottoTestF3 = ottoTestF3.add_suffix("_f3")


# In[8]:

#Engineered Feature4: scale X
ottoTrainScale = pd.DataFrame(scale(ottoX))
ottoTestScale  = pd.DataFrame(scale(ottoTestDat))

ottoTrainF4 = (ottoTrainScale > 0).astype(int).sum(axis=1)
ottoTrainF4 = ottoTrainF4.rename("newFeature4")
ottoTestF4  = (ottoTestScale > 0).astype(int).sum(axis=1)
ottoTestF4 = ottoTestF4.rename("newFeature4")


# In[66]:

#Define function for generating level-1 metafeatures (model predictions) using 4-fold CV
#for train and test dat.
def modelMetafeature(kfold, model, name) :
    #ref dat
    global ottoX
    global ottoY
    global ottoTestDat
    
    #init empty metafeature dataframe
    cur_train_MF = pd.DataFrame()
    cur_finalTest_MF  = 0
    
    #init k-fold index
    cv = KFold(n_splits= kfold, random_state=1)
    cvOttoX = cv.split(ottoX)
    
    for trainIx, testIx in cvOttoX:
        cur_X_train, cur_X_test = ottoX.ix[trainIx, :], ottoX.ix[testIx, :]
        cur_y_train, cur_y_test = ottoY[trainIx], ottoY[testIx]
        
        model.fit(cur_X_train, cur_y_train)
        
        cur_fold_pred = pd.DataFrame(model.predict_proba(cur_X_test),                                      index = testIx).add_suffix("_"+ name)
        cur_train_MF = cur_train_MF.append(cur_fold_pred)
        cur_fold_LL = log_loss(pd.get_dummies(cur_y_test), cur_fold_pred)
        print("Current testing fold LogLoss: ", cur_fold_LL)
        
        cur_finalTest_MF = np.add(cur_finalTest_MF, model.predict_proba(ottoTestDat)) 
    
    cur_train_MF.sort_index(inplace = True)
    cur_finalTest_MF = pd.DataFrame(cur_finalTest_MF / kfold).add_suffix("_" + name)
    
    return cur_train_MF, cur_finalTest_MF


# In[12]:

#Model Feature1: KNN = 8
knn8 = KNeighborsClassifier(n_neighbors=8)
ottoTrainMF1, ottoTestMF1 = modelMetafeature(4, knn8, "knn8")

#check dimension
print(ottoTrainMF1.shape)
print(ottoTestMF1.shape)


# In[15]:

#Model Feature2: KNN = 16
knn16 = KNeighborsClassifier(n_neighbors=16)
ottoTrainMF2, ottoTestMF2 = modelMetafeature(4, knn16, "knn16")

#check dimension
print(ottoTrainMF2.shape)
print(ottoTestMF2.shape)


# In[17]:

#Model Feature3: KNN = 32
knn32 = KNeighborsClassifier(n_neighbors=32)
ottoTrainMF3, ottoTestMF3 = modelMetafeature(4, knn32, "knn32")

#check dimension
print(ottoTrainMF3.shape)
print(ottoTestMF3.shape)


# In[19]:

#Model Feature4: KNN = 64
knn64 = KNeighborsClassifier(n_neighbors=64)
ottoTrainMF4, ottoTestMF4 = modelMetafeature(4, knn64, "knn64")

#check dimension
print(ottoTrainMF4.shape)
print(ottoTestMF4.shape)


# In[20]:

#Model Feature5: KNN = 128
knn128 = KNeighborsClassifier(n_neighbors=128)
ottoTrainMF5, ottoTestMF5 = modelMetafeature(4, knn128, "knn128")

#check dimension
print(ottoTrainMF5.shape)
print(ottoTestMF5.shape)


# In[22]:

#Model Feature6: RF
rf = RandomForestClassifier(n_estimators=700)
ottoTrainMF6, ottoTestMF6 = modelMetafeature(4, rf, "rf")

#check dimension
print(ottoTrainMF6.shape)
print(ottoTestMF6.shape)


# In[24]:

#Model Feature7: Logistic regression
logr = LogisticRegression()
ottoTrainMF7, ottoTestMF7 = modelMetafeature(4, logr, "logr")

#check dimension
print(ottoTrainMF7.shape)
print(ottoTestMF7.shape)


# In[26]:

#Model feature8: Extra Tree classifier
et800 = ExtraTreesClassifier(n_estimators=800)
ottoTrainMF8, ottoTestMF8 = modelMetafeature(4, et800, "et800")

#check dimension
print(ottoTrainMF8.shape)
print(ottoTestMF8.shape)


# In[28]:

#Model feature9: XBG learning rate = 0.1, tree = 600
xgb600 = xgboost.XGBClassifier(learning_rate = 0.1, max_depth = 4, n_estimators = 600)
ottoTrainMF9, ottoTestMF9 = modelMetafeature(4, xgb600, "xgb600")

#check dimension
print(ottoTrainMF9.shape)
print(ottoTestMF9.shape)


# In[30]:

#Model feature10: Neural Network: THREE layers
nn3L = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(20, 10, 10), random_state=1)
ottoTrainMF10, ottoTestMF10 = modelMetafeature(4, nn3L, "nn3L")

#check dimension
print(ottoTrainMF10.shape)
print(ottoTestMF10.shape)


# In[32]:

#Concat all new features as df(ottoXMeta) for train
ottoXMeta = pd.concat([ottoTrainMF1, ottoTrainMF2, ottoTrainMF3, ottoTrainMF4, ottoTrainMF5, \
    ottoTrainMF6, ottoTrainMF7, ottoTrainMF8, ottoTrainMF9, ottoTrainMF10, ottoTrainF1, \
    ottoTrainF2, ottoTrainF3, ottoTrainF4], axis = 1)


# In[33]:

#Level 2: Split train and test
xTrain, xTest, yTrain, yTest = ms.train_test_split(ottoXMeta, ottoY, test_size=0.3, random_state=1)

#Convert yTest to dummy
yTestDummy = pd.get_dummies(yTest).as_matrix()


# In[55]:

#Level 2: Model1: XBG learning rate = 0.1, tree = 300
xgbL2 = xgboost.XGBClassifier(learning_rate = 0.1, max_depth = 4, n_estimators = 600)
xgbL2.fit(xTrain, yTrain)
xgbL2Pred = xgbL2.predict_proba(xTest)
xgbL2LL = log_loss(yTestDummy, xgbL2Pred)
print(xgbL2LL)


# In[34]:

#Level 2: Model2, Neural Network: THREE layers
nnL2 = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(20, 10, 10), random_state=1)
nnL2.fit(xTrain, yTrain)
nnL2Pred = nnL2.predict_proba(xTest)
nnL2LL   = log_loss(yTestDummy, nnL2Pred)
print(nnL2LL)


# In[36]:

#Level 2: Model3, Extra Tree Clssifier
etL2 = ExtraTreesClassifier(n_estimators=900)
etL2.fit(xTrain, yTrain)
etL2Pred = etL2.predict_proba(xTest)
etL2LL = log_loss(yTestDummy, etL2Pred)
print(etL2LL)


# In[37]:

#Level 2: Ensemble1
ensemble1 = 0.75*xgbL2Pred + 0.15*nnL2Pred + 0.1*etL2Pred
ensemble1LL = log_loss(yTestDummy, ensemble1)
print(ensemble1LL)


# In[38]:

#Level 2: Ensemble2
ensemble2 = (xgbL2Pred**0.5)*(nnL2Pred**0.25)*(etL2Pred**0.25)
ensemble2LL = log_loss(yTestDummy, ensemble2)
print(ensemble2LL)


# In[51]:

#Level 2: Ensemble3
ensemble3 = 0.6*((xgbL2Pred**0.65)*(nnL2Pred**0.35))+0.4*etL2Pred
ensemble3LL = log_loss(yTestDummy, ensemble3)
print(ensemble3LL)


# In[53]:

#Level 3: Predict test dat
#Concat all new features as df(ottoXTestMeta)
ottoXTestMeta = pd.concat([ottoTestMF1, ottoTestMF2, ottoTestMF3, ottoTestMF4, ottoTestMF5, \
                        ottoTestMF6, ottoTestMF7, ottoTestMF8, ottoTestMF9, ottoTestMF10, ottoTestF1, \
                        ottoTestF2, ottoTestF3, ottoTestF4], axis = 1)


# In[54]:

#Level 3: Select final model and write result
xgbPred = xgbL2.predict_proba(ottoXTestMeta)
nnPred  = nnL2.predict_proba(ottoXTestMeta)
etPred  = etL2.predict_proba(ottoXTestMeta)
ensPred = 0.6*((xgbPred**0.65)*(nnPred**0.35)) + 0.4*etPred
ensPredPd = pd.DataFrame(ensPred)
ensFinal = pd.concat([ottoTestId, ensPredPd], axis=1)
ensFinal.to_csv("/Users/JaneShi/Desktop/MSCA31009/Project/ensXgbNNET_Final.csv", header=False, index= False)

