from matplotlib import pyplot as plt
import csv
import pandas as pd
from numpy import genfromtxt
import numpy as np
import os

populationSize = [100]
numOfElites = [4, 8, 16]
mutationRate = [0.05, 0.10, 0.20]
maxAbs = [0.1, 0.4, 0.8]

interia = [0.0, 0.37, 0.74]
personal = [0.5, 1, 1.5]
social = [0.5, 1, 1.5]

no_dims=[5,10]

parentFile="C:/Fakultet/FERIT/8. semestar/Meko računarstvo/LV/LV3/GA"

def plot(data1, data2, data3, legend, title):
    plt.figure()
    plt.plot(data1[:,0], -data1[:,1], c="red")
    plt.plot(data2[:,0], -data2[:,1], c="green")
    plt.plot(data3[:,0], -data3[:,1], c="blue")
    plt.xscale("log")
    plt.xlabel("Generacija")
    plt.ylabel("Fitness")
    plt.legend(legend)
    plt.title(title)
    filePath= parentFile + "/graphPlot/" + title + ".png"
    plt.savefig(filePath)
    
#GA
#5_Dimensions&0.1_mutationProb&4_numOfElites&0.1_maxMutatedValue_best

#PSO
#5_Dimensions&0.0_InertiaVal&0.5_individualFactor&0.5_socialFactor_median


#Changing mutation rate

filePath1_5= parentFile + "/graphData/" + str(no_dims[0]) + "_Dimensions&" + str(mutationRate[0]) + "_mutationProb&" + str(numOfElites[0]) + "_numOfElites&" + str(maxAbs[0]) + "_maxMutatedValue_median.csv"
filePath2_5= parentFile + "/graphData/" + str(no_dims[0]) + "_Dimensions&" + str(mutationRate[1]) + "_mutationProb&" + str(numOfElites[0]) + "_numOfElites&" + str(maxAbs[0]) + "_maxMutatedValue_median.csv"
filePath3_5= parentFile + "/graphData/" + str(no_dims[0]) + "_Dimensions&" + str(mutationRate[2]) + "_mutationProb&" + str(numOfElites[0]) + "_numOfElites&" + str(maxAbs[0]) + "_maxMutatedValue_median.csv"

filePath1_1= parentFile + "/graphData/" + str(no_dims[1]) + "_Dimensions&" + str(mutationRate[0]) + "_mutationProb&" + str(numOfElites[0]) + "_numOfElites&" + str(maxAbs[0]) + "_maxMutatedValue_median.csv"
filePath2_1= parentFile + "/graphData/" + str(no_dims[1]) + "_Dimensions&" + str(mutationRate[1]) + "_mutationProb&" + str(numOfElites[0]) + "_numOfElites&" + str(maxAbs[0]) + "_maxMutatedValue_median.csv"
filePath3_1= parentFile + "/graphData/" + str(no_dims[1]) + "_Dimensions&" + str(mutationRate[2]) + "_mutationProb&" + str(numOfElites[0]) + "_numOfElites&" + str(maxAbs[0]) + "_maxMutatedValue_median.csv"

data1_T = genfromtxt(filePath1_5, delimiter=',')
data2_T = genfromtxt(filePath2_5, delimiter=',')
data3_T = genfromtxt(filePath3_5, delimiter=',')

data1_F = genfromtxt(filePath1_1, delimiter=',')
data2_F = genfromtxt(filePath2_1, delimiter=',')
data3_F = genfromtxt(filePath3_1, delimiter=',')

plot(data1_T, data2_T, data3_T, ["Mutation rate = 5%", "Mutation rate = 10%", "Mutation rate = 20%"], "Ovisnost o postotku mutacije za dimenziju 5")

plot(data1_F, data2_F, data3_F, ["Mutation rate = 5%", "Mutation rate = 10%", "Mutation rate = 20%"], "Ovisnost o postotku mutacije za dimenziju 10")

print("Saved the plotted files")



#Changing number of elites

filePath1_5= parentFile + "/graphData/" + str(no_dims[0]) + "_Dimensions&" + str(mutationRate[0]) + "_mutationProb&" + str(numOfElites[0]) + "_numOfElites&" + str(maxAbs[0]) + "_maxMutatedValue_median.csv"
filePath2_5= parentFile + "/graphData/" + str(no_dims[0]) + "_Dimensions&" + str(mutationRate[0]) + "_mutationProb&" + str(numOfElites[1]) + "_numOfElites&" + str(maxAbs[0]) + "_maxMutatedValue_median.csv"
filePath3_5= parentFile + "/graphData/" + str(no_dims[0]) + "_Dimensions&" + str(mutationRate[0]) + "_mutationProb&" + str(numOfElites[2]) + "_numOfElites&" + str(maxAbs[0]) + "_maxMutatedValue_median.csv"

filePath1_1= parentFile + "/graphData/" + str(no_dims[1]) + "_Dimensions&" + str(mutationRate[0]) + "_mutationProb&" + str(numOfElites[0]) + "_numOfElites&" + str(maxAbs[0]) + "_maxMutatedValue_median.csv"
filePath2_1= parentFile + "/graphData/" + str(no_dims[1]) + "_Dimensions&" + str(mutationRate[0]) + "_mutationProb&" + str(numOfElites[1]) + "_numOfElites&" + str(maxAbs[0]) + "_maxMutatedValue_median.csv"
filePath3_1= parentFile + "/graphData/" + str(no_dims[1]) + "_Dimensions&" + str(mutationRate[0]) + "_mutationProb&" + str(numOfElites[2]) + "_numOfElites&" + str(maxAbs[0]) + "_maxMutatedValue_median.csv"

data1_T = genfromtxt(filePath1_5, delimiter=',')
data2_T = genfromtxt(filePath2_5, delimiter=',')
data3_T = genfromtxt(filePath3_5, delimiter=',')

data1_F = genfromtxt(filePath1_1, delimiter=',')
data2_F = genfromtxt(filePath2_1, delimiter=',')
data3_F = genfromtxt(filePath3_1, delimiter=',')

plot(data1_T, data2_T, data3_T, ["Broj elitnih članova = 4", "Broj elitnih članova = 8", "Broj elitnih članova = 16"], "Ovisnost o broju elitnih članova za dimenziju 5")

plot(data1_F, data2_F, data3_F, ["Broj elitnih članova = 4", "Broj elitnih članova = 8", "Broj elitnih članova = 16"], "Ovisnost o broju elitnih članova za dimenziju 10")

print("Saved the plotted files")



#Changing maximum absolute value of mutation of the real gene

filePath1_5= parentFile + "/graphData/" + str(no_dims[0]) + "_Dimensions&" + str(mutationRate[0]) + "_mutationProb&" + str(numOfElites[0]) + "_numOfElites&" + str(maxAbs[0]) + "_maxMutatedValue_median.csv"
filePath2_5= parentFile + "/graphData/" + str(no_dims[0]) + "_Dimensions&" + str(mutationRate[0]) + "_mutationProb&" + str(numOfElites[0]) + "_numOfElites&" + str(maxAbs[1]) + "_maxMutatedValue_median.csv"
filePath3_5= parentFile + "/graphData/" + str(no_dims[0]) + "_Dimensions&" + str(mutationRate[0]) + "_mutationProb&" + str(numOfElites[0]) + "_numOfElites&" + str(maxAbs[2]) + "_maxMutatedValue_median.csv"

filePath1_1= parentFile + "/graphData/" + str(no_dims[1]) + "_Dimensions&" + str(mutationRate[0]) + "_mutationProb&" + str(numOfElites[0]) + "_numOfElites&" + str(maxAbs[0]) + "_maxMutatedValue_median.csv"
filePath2_1= parentFile + "/graphData/" + str(no_dims[1]) + "_Dimensions&" + str(mutationRate[0]) + "_mutationProb&" + str(numOfElites[0]) + "_numOfElites&" + str(maxAbs[1]) + "_maxMutatedValue_median.csv"
filePath3_1= parentFile + "/graphData/" + str(no_dims[1]) + "_Dimensions&" + str(mutationRate[0]) + "_mutationProb&" + str(numOfElites[0]) + "_numOfElites&" + str(maxAbs[2]) + "_maxMutatedValue_median.csv"

data1_T = genfromtxt(filePath1_5, delimiter=',')
data2_T = genfromtxt(filePath2_5, delimiter=',')
data3_T = genfromtxt(filePath3_5, delimiter=',')

data1_F = genfromtxt(filePath1_1, delimiter=',')
data2_F = genfromtxt(filePath2_1, delimiter=',')
data3_F = genfromtxt(filePath3_1, delimiter=',')

plot(data1_T, data2_T, data3_T, ["Najveća apsolutna vrijednost mutacije realnog gena = 0.1", "Najveća apsolutna vrijednost mutacije realnog gena = 0.4", "Najveća apsolutna vrijednost mutacije realnog gena = 0.8"], "Ovisnost o vrijednosti mutacije realnog gena za dimenziju 5")

plot(data1_F, data2_F, data3_F, ["Najveća apsolutna vrijednost mutacije realnog gena = 0.1", "Najveća apsolutna vrijednost mutacije realnog gena = 0.4", "Najveća apsolutna vrijednost mutacije realnog gena = 0.8"], "Ovisnost o vrijednosti mutacije realnog gena za dimenziju 10")

print("Saved the plotted files")
