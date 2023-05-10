from matplotlib import pyplot as plt
import csv
import pandas as pd
from numpy import genfromtxt
import numpy as np
import os


populationSize = [50,100,200,400]
numOfElites = [5,10,15,20]
mutationRate = [0.05, 0.10, 0.15, 0.20]
borderPatrol=[True,False]

parentFile="C:\Fakultet\FERIT\8. semestar\Meko računarstvo\LV\LV2"


def plot(data1, data2, data3, data4, legend, title):
    x=np.asarray(list(range(0,5000)))

    fitnessValues1=[]
    fitnessValues2=[]
    fitnessValues3=[]
    fitnessValues4=[]

    for i in range(4):
        fitnessValues1.append(data1[i,-1])
        fitnessValues2.append(data2[i,-1])
        fitnessValues3.append(data3[i,-1])
        fitnessValues4.append(data4[i,-1])

    minIndex1=np.argmin(fitnessValues1)
    minIndex2=np.argmin(fitnessValues2)
    minIndex3=np.argmin(fitnessValues3)
    minIndex4=np.argmin(fitnessValues4)

    plt.figure()
    plt.plot(x, -data1[minIndex1, 1:], c="red")
    plt.plot(x, -data2[minIndex2, 1:], c="green")
    plt.plot(x, -data3[minIndex3, 1:], c="blue")
    plt.plot(x, -data4[minIndex4, 1:], c="black")
    plt.xlabel("Generacija")
    plt.ylabel("Fitness")
    plt.legend(legend)
    plt.title(title)
    filePath= parentFile + "\graphPlot\_" + title + ".png"
    plt.savefig(filePath)
    

#BorderPatrol=False_popSize=50_numElites=5_mutationRate=0.1

#Changing population

filePath1_T= parentFile + "\graphData\BorderPatrol=" + str(borderPatrol[0]) + "_popSize=" + str(populationSize[0]) + "_numElites=" + str(numOfElites[0]) + "_mutationRate=" + str(mutationRate[0]) + ".csv"
filePath2_T= parentFile + "\graphData\BorderPatrol=" + str(borderPatrol[0]) + "_popSize=" + str(populationSize[1]) + "_numElites=" + str(numOfElites[0]) + "_mutationRate=" + str(mutationRate[0]) + ".csv"
filePath3_T= parentFile + "\graphData\BorderPatrol=" + str(borderPatrol[0]) + "_popSize=" + str(populationSize[2]) + "_numElites=" + str(numOfElites[0]) + "_mutationRate=" + str(mutationRate[0]) + ".csv"
filePath4_T= parentFile + "\graphData\BorderPatrol=" + str(borderPatrol[0]) + "_popSize=" + str(populationSize[3]) + "_numElites=" + str(numOfElites[0]) + "_mutationRate=" + str(mutationRate[0]) + ".csv"

filePath1_F= parentFile + "\graphData\BorderPatrol=" + str(borderPatrol[1]) + "_popSize=" + str(populationSize[0]) + "_numElites=" + str(numOfElites[0]) + "_mutationRate=" + str(mutationRate[0]) + ".csv"
filePath2_F= parentFile + "\graphData\BorderPatrol=" + str(borderPatrol[1]) + "_popSize=" + str(populationSize[1]) + "_numElites=" + str(numOfElites[0]) + "_mutationRate=" + str(mutationRate[0]) + ".csv"
filePath3_F= parentFile + "\graphData\BorderPatrol=" + str(borderPatrol[1]) + "_popSize=" + str(populationSize[2]) + "_numElites=" + str(numOfElites[0]) + "_mutationRate=" + str(mutationRate[0]) + ".csv"
filePath4_F= parentFile + "\graphData\BorderPatrol=" + str(borderPatrol[1]) + "_popSize=" + str(populationSize[3]) + "_numElites=" + str(numOfElites[0]) + "_mutationRate=" + str(mutationRate[0]) + ".csv"

data1_T = genfromtxt(filePath1_T, delimiter=',')
data2_T = genfromtxt(filePath2_T, delimiter=',')
data3_T = genfromtxt(filePath3_T, delimiter=',')
data4_T = genfromtxt(filePath4_T, delimiter=',')

data1_F = genfromtxt(filePath1_F, delimiter=',')
data2_F = genfromtxt(filePath2_F, delimiter=',')
data3_F = genfromtxt(filePath3_F, delimiter=',')
data4_F = genfromtxt(filePath4_F, delimiter=',')

plot(data1_T, data2_T, data3_T, data4_T, ["Populacija=50", "Populacija=100", "Populacija=150", "Populacija=200"], "Ovisnost o veličini populacije za border patrol = True")

plot(data1_F, data2_F, data3_F, data4_F, ["Populacija=50", "Populacija=100", "Populacija=150", "Populacija=200"], "Ovisnost o veličini populacije za border patrol = False")

print("Saved the plotted files")



#Changing mutation rate

filePath1_T= parentFile + "\graphData\BorderPatrol=" + str(borderPatrol[0]) + "_popSize=" + str(populationSize[0]) + "_numElites=" + str(numOfElites[0]) + "_mutationRate=" + str(mutationRate[0]) + ".csv"
filePath2_T= parentFile + "\graphData\BorderPatrol=" + str(borderPatrol[0]) + "_popSize=" + str(populationSize[0]) + "_numElites=" + str(numOfElites[0]) + "_mutationRate=" + str(mutationRate[1]) + ".csv"
filePath3_T= parentFile + "\graphData\BorderPatrol=" + str(borderPatrol[0]) + "_popSize=" + str(populationSize[0]) + "_numElites=" + str(numOfElites[0]) + "_mutationRate=" + str(mutationRate[2]) + ".csv"
filePath4_T= parentFile + "\graphData\BorderPatrol=" + str(borderPatrol[0]) + "_popSize=" + str(populationSize[0]) + "_numElites=" + str(numOfElites[0]) + "_mutationRate=" + str(mutationRate[3]) + ".csv"

filePath1_F= parentFile + "\graphData\BorderPatrol=" + str(borderPatrol[1]) + "_popSize=" + str(populationSize[0]) + "_numElites=" + str(numOfElites[0]) + "_mutationRate=" + str(mutationRate[0]) + ".csv"
filePath2_F= parentFile + "\graphData\BorderPatrol=" + str(borderPatrol[1]) + "_popSize=" + str(populationSize[0]) + "_numElites=" + str(numOfElites[0]) + "_mutationRate=" + str(mutationRate[1]) + ".csv"
filePath3_F= parentFile + "\graphData\BorderPatrol=" + str(borderPatrol[1]) + "_popSize=" + str(populationSize[0]) + "_numElites=" + str(numOfElites[0]) + "_mutationRate=" + str(mutationRate[2]) + ".csv"
filePath4_F= parentFile + "\graphData\BorderPatrol=" + str(borderPatrol[1]) + "_popSize=" + str(populationSize[0]) + "_numElites=" + str(numOfElites[0]) + "_mutationRate=" + str(mutationRate[3]) + ".csv"

data1_T = genfromtxt(filePath1_T, delimiter=',')
data2_T = genfromtxt(filePath2_T, delimiter=',')
data3_T = genfromtxt(filePath3_T, delimiter=',')
data4_T = genfromtxt(filePath4_T, delimiter=',')

data1_F = genfromtxt(filePath1_F, delimiter=',')
data2_F = genfromtxt(filePath2_F, delimiter=',')
data3_F = genfromtxt(filePath3_F, delimiter=',')
data4_F = genfromtxt(filePath4_F, delimiter=',')

plot(data1_T, data2_T, data3_T, data4_T, ["Mutation rate=5%" , "Mutation rate=10%", "Mutation rate=15%" , "Mutation rate=20%"], "Ovisnost o postotku mutacije za border patrol = True")

plot(data1_F, data2_F, data3_F, data4_F, ["Mutation rate=5%" , "Mutation rate=10%", "Mutation rate=15%" , "Mutation rate=20%"], "Ovisnost o postotku mutacije za border patrol = False")

print("Saved the plotted files")



#Changing number of elites

filePath1_T= parentFile + "\graphData\BorderPatrol=" + str(borderPatrol[0]) + "_popSize=" + str(populationSize[0]) + "_numElites=" + str(numOfElites[0]) + "_mutationRate=" + str(mutationRate[0]) + ".csv"
filePath2_T= parentFile + "\graphData\BorderPatrol=" + str(borderPatrol[0]) + "_popSize=" + str(populationSize[0]) + "_numElites=" + str(numOfElites[1]) + "_mutationRate=" + str(mutationRate[0]) + ".csv"
filePath3_T= parentFile + "\graphData\BorderPatrol=" + str(borderPatrol[0]) + "_popSize=" + str(populationSize[0]) + "_numElites=" + str(numOfElites[2]) + "_mutationRate=" + str(mutationRate[0]) + ".csv"
filePath4_T= parentFile + "\graphData\BorderPatrol=" + str(borderPatrol[0]) + "_popSize=" + str(populationSize[0]) + "_numElites=" + str(numOfElites[3]) + "_mutationRate=" + str(mutationRate[0]) + ".csv"

filePath1_F= parentFile + "\graphData\BorderPatrol=" + str(borderPatrol[1]) + "_popSize=" + str(populationSize[0]) + "_numElites=" + str(numOfElites[0]) + "_mutationRate=" + str(mutationRate[0]) + ".csv"
filePath2_F= parentFile + "\graphData\BorderPatrol=" + str(borderPatrol[1]) + "_popSize=" + str(populationSize[0]) + "_numElites=" + str(numOfElites[1]) + "_mutationRate=" + str(mutationRate[0]) + ".csv"
filePath3_F= parentFile + "\graphData\BorderPatrol=" + str(borderPatrol[1]) + "_popSize=" + str(populationSize[0]) + "_numElites=" + str(numOfElites[2]) + "_mutationRate=" + str(mutationRate[0]) + ".csv"
filePath4_F= parentFile + "\graphData\BorderPatrol=" + str(borderPatrol[1]) + "_popSize=" + str(populationSize[0]) + "_numElites=" + str(numOfElites[3]) + "_mutationRate=" + str(mutationRate[0]) + ".csv"

data1_T = genfromtxt(filePath1_T, delimiter=',')
data2_T = genfromtxt(filePath2_T, delimiter=',')
data3_T = genfromtxt(filePath3_T, delimiter=',')
data4_T = genfromtxt(filePath4_T, delimiter=',')

data1_F = genfromtxt(filePath1_F, delimiter=',')
data2_F = genfromtxt(filePath2_F, delimiter=',')
data3_F = genfromtxt(filePath3_F, delimiter=',')
data4_F = genfromtxt(filePath4_F, delimiter=',')

plot(data1_T, data2_T, data3_T, data4_T, ["Broj elita=5" , "Broj elita=10", "Broj elita=15" , "Broj elita=20"], "Ovisnost o broju elita za border patrol = True")

plot(data1_F, data2_F, data3_F, data4_F, ["Broj elita=5" , "Broj elita=10", "Broj elita=15" , "Broj elita=20"], "Ovisnost o broju elita za border patrol = False")

print("Saved the plotted files")