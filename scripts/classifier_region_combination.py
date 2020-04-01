#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 15:09:50 2020

@author: izaiasjr
"""

import pandas as pd  
import numpy as np  

from sklearn.preprocessing import PowerTransformer

from sklearn.svm import LinearSVC
from sklearn.neighbors import NearestCentroid as NC
from sklearn.neighbors import KNeighborsClassifier as KNN

from sklearn.metrics import accuracy_score #works


result={}
norm_rays=[15,20,25,30,35,40]
for norm_radius in norm_rays:
  
  path='features-extraction'
  extractor='FPFH/norm_{}'.format(norm_radius)
  #extractor='FPFH'
  
  dim=33
  
  nose_tip = pd.read_csv("{}/{}/{}/{}".format(path,extractor,'nose_tip','nose_tip_30_FPFH.dat'))
  mouth_l =  pd.read_csv("{}/{}/{}/{}".format(path,extractor,'mouth_l','mouth_l_50_FPFH.dat'))
  mouth_r =  pd.read_csv("{}/{}/{}/{}".format(path,extractor,'mouth_r','mouth_r_50_FPFH.dat'))
  mouth_cu = pd.read_csv("{}/{}/{}/{}".format(path,extractor,'mouth_cu','mouth_cu_40_FPFH.dat'))
  mouth_cd = pd.read_csv("{}/{}/{}/{}".format(path,extractor,'mouth_cd','mouth_cd_40_FPFH.dat'))
  eye_le = pd.read_csv("{}/{}/{}/{}".format(path,extractor,'eye_le','eye_le_50_FPFH.dat'))
  eye_re = pd.read_csv("{}/{}/{}/{}".format(path,extractor,'eye_re','eye_re_50_FPFH.dat'))
  eye_li = pd.read_csv("{}/{}/{}/{}".format(path,extractor,'eye_li','eye_li_50_FPFH.dat'))
  eye_ri = pd.read_csv("{}/{}/{}/{}".format(path,extractor,'eye_ri','eye_ri_40_FPFH.dat'))
  
  
   
  infos=nose_tip.loc[:,'subject':]
  
  # concatenation
  df = pd.concat([
                    nose_tip.iloc[:,:dim],
                    mouth_l.iloc[:,:dim],
                    mouth_r.iloc[:,:dim],
                    mouth_cu.iloc[:,:dim],
                    mouth_cd.iloc[:,:dim],
                    eye_le.iloc[:,:dim],
                    eye_re.iloc[:,:dim],
                    eye_ri.iloc[:,:dim],
                    eye_li.iloc[:,:dim],
                    infos
                    ],axis=1) 
  
  #df = pd.read_csv("features-extraction/concat6_FPFH.dat")
  
  
#  cond_train = ( ( df["tp"] == "N") & ((df["sample"] == 0) | (df["sample"] == 1) ) ) 
  cond_train = ( ( df["tp"] == "N") & ((df["sample"] == 0) ) )
  trainset = df.loc[cond_train].drop(["sample", "tp", "exp"], axis=1)
  X_train = np.array(trainset.drop(["subject"], axis=1))
  y_train = np.ravel(trainset[["subject"]])
  
#  cond_test = ( (df["tp"] == "N") & ((df["sample"] !=0) | (df["sample"] !=1)))
#  cond_test = ( (df["tp"] == "O") & (df["exp"] == 'HAIR') )
  cond_test = ( (df["tp"] == "N") & (df["sample"] !=0)  )
  testset = df.loc[cond_test].drop(["sample", "tp", "exp"], axis=1)
  X_test = np.array(testset.drop(["subject"], axis=1))
  y_test = np.ravel(testset[["subject"]])
  
  ## PRE-PROCESSING    
  scaler = PowerTransformer().fit(X_train)
  X_train = scaler.transform(X_train)
  X_test = scaler.transform(X_test)
  # remove NaN
  #X_test=np.nan_to_num(X_test)
  #X_train=np.nan_to_num(X_train)
  
  
  
  classifiers={
#       "LDA":LDA(),
  #     "QDA": QDA(), 
#       "MLP":MLPClassifier(alpha=2, max_iter=1000),
#       "DecisionTree":DecisionTreeClassifier(), 
#        "GaussianNB":GaussianNB(),
#        "RandomForest":RandomForestClassifier(n_estimators=1000),
        "KNN-manhattan":KNN(metric="manhattan", n_neighbors=1),
        "KNN-euclidean":KNN(metric="euclidean", n_neighbors=1),
        "LinearSVC":LinearSVC(),      
#        "KNN-minkowski-3":KNN(metric='minkowski', p=3, n_neighbors=1),
#        "KNN-minkowski-4":KNN(metric='minkowski', p=4, n_neighbors=1),        
        
      } 
  
  accuracies= []
  for label_classifier in classifiers:
    

      
    ## CLASSIFICATION
    classifier=classifiers[label_classifier]
    classifier.fit(X_train, y_train)  
    y_pred = classifier.predict(X_test)  
     
    accur = round(100*accuracy_score(y_test, y_pred),2)
    accuracies.append({label_classifier:accur})
    
    if label_classifier not in result:
      result[label_classifier]=[]
    result[label_classifier].append(accur)
  
  
#  result.append(accuracies)
for r in result: 
  plt.plot(norm_rays,result[r])
plt.legend(loc='best', shadow=True, fontsize='x-large')
