#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 09:12:42 2019

@author: nicolas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 

#Chargement du dataset
def ChargementDataset(lien):
    data = pd.read_csv(lien, header = None , sep =" ")
    return data

#Calcul de la distance euclidienne
def distanceEuclidienne(point1, point2,taille):
    distance =0
    for x in range(taille):
        distance += np.square(point1[x]-point2[x])
    return np.sqrt(distance)

#Implémentation de notre modèle(renvoie les classes des element le plus proche de la data de test)
def knn_algo(training,test,k):
    distances = {}
    taille = (len(test)-2)
    #Distance euclidienne entre la donnée de test et tout le dataset de train
    for x in range(len(training)):
        dist = distanceEuclidienne(test, training.iloc[x], taille)
        train = training.iloc[x] 
        distances[dist] =  train[len(train)-1]
           
    # Trier le dictionnaire de distances 
    sorted_d = sorted(distances.items(),key=lambda t: t[0])
    voisin = []
    #Extraction des éléments les plus proche en fonction du k
    for x in range(k):
        voisin.append(sorted_d[x][1])
    return voisin

def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    plt.colorbar()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)


def main():
    print("**********************************************")
    print("*                                            *")
    print("*           IMPLEMENTATION DE KNN            *")
    print("*                                            *")
    print("*                                            *")
    print("**********************************************")
    print("                                              ")
    lienTrain = input("Entrer le lien de train ")
    lienTest = input("Entrer le lien de test ")
    k = int(input("Entrer le K "))
    dataTrain = ChargementDataset(lienTrain)
    dataTest = ChargementDataset(lienTest)
    
    y_test = []
    y_predi = []
    for test in range(len(dataTest)):
        k_PP = (knn_algo(dataTrain,dataTest.iloc[test],k))
        y_test.append(dataTest.iloc[test][len(dataTest.iloc[test])-1])
        y_predi.append(max(k_PP,key=k_PP.count))
        
        
    y_actu = pd.Series(y_test, name='Actuel')
    y_pred = pd.Series(y_predi, name='Prediction')
    df_confusion = pd.crosstab(y_actu, y_pred)
    print(df_confusion)
    plot_confusion_matrix(df_confusion)
if __name__ == '__main__':
    main()
    
 
  




