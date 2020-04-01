#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np  
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.svm import LinearSVC

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

from sklearn.neighbors import NearestCentroid as NC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


from sklearn.metrics import * #works
import matplotlib.pyplot as plt 
from matplotlib.pyplot import figure

def plotAccuracia(norm_radius,region,classifiers,toSave): 

  #fig, ax = plt.subplots()
  plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
  plt.xlabel('Raios (mm)')
  plt.ylabel('Accurácia (%)')
  plt.title('{} - norm:{}'.format(region,norm_radius),size=20)
  plt.grid(True)
    
  idx=0
  for result_classifiers in classifiers:
    idx=idx+1
    r_accur = results[result_classifiers]['accuracies']
    plt.plot(rays_feat,r_accur,
             marker='*',label='{}- {}%'.format(result_classifiers,max(r_accur)))
    
  #   ax.plot(rays_feat,results[result_classifiers]['accuracies'], 1) 
  plt.legend(loc='best', shadow=True, fontsize='x-large')
 
  if toSave == True:
    plt.savefig('./results/{}.png'.format(region))
  

def plotAccuraciaRegions(results_regions): 
    
  ## Config plot
  fig, axs = plt.subplots(1 ,3, figsize=(9, 6),dpi=80, facecolor='w', edgecolor='k')
  fig.subplots_adjust(hspace = .5, wspace=.1)
  axs = axs.ravel()
  
  ## iterate over infos
  i=0
  for region in results_regions:
    for classifier in  results_regions[region]:
      accur=results_regions[region][classifier]['accuracies']
      rays=results_regions[region][classifier]['rays']
      axs[i].plot(rays,accur, label='{}'.format(classifier))   
    i=i+1  
    
  # legend
  handles, labels = axs[i].get_legend_handles_labels()
  fig.legend(handles, labels, loc='upper center')
  

#norm_rays=[15,20,25,30,35,40] 
norm_rays=[15]
for norm_radius in norm_rays:
  classifiers={
  #     "LDA":LDA(),
  #     "QDA": QDA(), 
  #     "MLP":MLPClassifier(alpha=2, max_iter=1000),
  #     "DecisionTree":DecisionTreeClassifier(), 
  #      "GaussianNB":GaussianNB(),
  #      "RandomForest":RandomForestClassifier(n_estimators=1000),
        "LinearSVC":LinearSVC(),      
        "NC-Euclidean":NC(metric="euclidean"),
        "NC-Manhattam":NC(metric="manhattan"),
        "KNN3":KNN(p=3, n_neighbors=1),
        "KNN4":KNN(p=4, n_neighbors=1),
      } 
  
  regions = ["nose_tip", "eye_ri", "eye_re", "eye_li","eye_le", "mouth_r", "mouth_l", "mouth_cu", "mouth_cd"]
  
  results_regions={}
  for region in regions:
    extractor='FPFH'
    results={}
    for label_classifier in classifiers:
          
    #  rays_feat=[5,10, 15, 20, 25, 30, 35, 40, 45, 50]
      rays_feat=[ 30, 40, 50]
      accuracies=[]
      for radius_feat in rays_feat:
      
        # OPEN  DATA
        filename='{}_{}_{}.dat'.format(region,radius_feat, extractor)
        pathFile="features-extraction/oclusions/{}/norm_{}/{}/{}".format(extractor,norm_radius,region,filename)
        df = pd.read_csv(pathFile)
        #df = pd.read_csv("features-extraction/concat6_FPFH.dat")
        
  #      cond_train = ( ( df["tp"] == "N") & ((df["sample"] == 0) | (df["sample"] == 1) ) ) 
        cond_train = ( ( df["tp"] == "N") & ((df["sample"] == 0) ) )
        trainset = df.loc[cond_train].drop(["sample", "tp", "exp"], axis=1)
        X_train = np.array(trainset.drop(["subject"], axis=1))
        y_train = np.ravel(trainset[["subject"]])
      
  #      cond_test = ( (df["tp"] == "N") & ((df["sample"] !=0) | (df["sample"] !=1)))
        cond_test = ( (df["tp"] == "O") & (df["exp"] == 'GLASSES')  )
#        cond_test = ( (df["tp"] == "N") & (df["sample"] !=0)  )
        testset = df.loc[cond_test].drop(["sample", "tp", "exp"], axis=1)
        X_test = np.array(testset.drop(["subject"], axis=1))
        y_test = np.ravel(testset[["subject"]])
        
        ## PRE-PROCESSING    
        scaler = PowerTransformer().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        # remove NaN
    #    X_test=np.nan_to_num(X_test)
    #    X_train=np.nan_to_num(X_train)
    #    # handle infinity
    
        ## CLASSIFICATION
        classifier=classifiers[label_classifier]
        classifier.fit(X_train, y_train)  
        y_pred = classifier.predict(X_test)  
        
        accuracies.append(round(100*accuracy_score(y_test, y_pred),2))
        
      results[label_classifier] = {'accuracies':accuracies,'rays':rays_feat}
    
    results_regions[region] = results
   
    plotAccuracia(norm_radius,region,classifiers,False)
  
