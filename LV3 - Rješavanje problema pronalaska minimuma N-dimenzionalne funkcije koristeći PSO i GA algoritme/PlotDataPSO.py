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

parentFile="C:/Fakultet/FERIT/8. semestar/Meko računarstvo/LV/LV3/PSO"

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
#5_Dimensions&0.0_InertiaVal&0.5_individualFactor&0.5_socialFactor_median

#Changing inertia value

filePath1_5= parentFile + "/graphData/" + str(no_dims[0]) + "_Dimensions&" + str(interia[0]) + "_InertiaVal&" + str(personal[0]) + "_individualFactor&" + str(social[0]) + "_socialFactor_median.csv"
filePath2_5= parentFile + "/graphData/" + str(no_dims[0]) + "_Dimensions&" + str(interia[1]) + "_InertiaVal&" + str(personal[0]) + "_individualFactor&" + str(social[0]) + "_socialFactor_median.csv"
filePath3_5= parentFile + "/graphData/" + str(no_dims[0]) + "_Dimensions&" + str(interia[2]) + "_InertiaVal&" + str(personal[0]) + "_individualFactor&" + str(social[0]) + "_socialFactor_median.csv"

filePath1_1= parentFile + "/graphData/" + str(no_dims[1]) + "_Dimensions&" + str(interia[0]) + "_InertiaVal&" + str(personal[0]) + "_individualFactor&" + str(social[0]) + "_socialFactor_median.csv"
filePath2_1= parentFile + "/graphData/" + str(no_dims[1]) + "_Dimensions&" + str(interia[1]) + "_InertiaVal&" + str(personal[0]) + "_individualFactor&" + str(social[0]) + "_socialFactor_median.csv"
filePath3_1= parentFile + "/graphData/" + str(no_dims[1]) + "_Dimensions&" + str(interia[2]) + "_InertiaVal&" + str(personal[0]) + "_individualFactor&" + str(social[0]) + "_socialFactor_median.csv"

data1_T = genfromtxt(filePath1_5, delimiter=',')
data2_T = genfromtxt(filePath2_5, delimiter=',')
data3_T = genfromtxt(filePath3_5, delimiter=',')

data1_F = genfromtxt(filePath1_1, delimiter=',')
data2_F = genfromtxt(filePath2_1, delimiter=',')
data3_F = genfromtxt(filePath3_1, delimiter=',')

plot(data1_T, data2_T, data3_T, ["Mjera inercije = 0.0", "Mjera inercije = 0.37", "Mjera inercije = 0.74"], "Ovisnost o mjeri inercije za dimenziju 5")

plot(data1_F, data2_F, data3_F, ["Mjera inercije = 0.0", "Mjera inercije = 0.37", "Mjera inercije = 0.74"], "Ovisnost o mjeri inercije za dimenziju 10")

print("Saved the plotted files")



#Changing personal value

filePath1_5= parentFile + "/graphData/" + str(no_dims[0]) + "_Dimensions&" + str(interia[0]) + "_InertiaVal&" + str(personal[0]) + "_individualFactor&" + str(social[0]) + "_socialFactor_median.csv"
filePath2_5= parentFile + "/graphData/" + str(no_dims[0]) + "_Dimensions&" + str(interia[0]) + "_InertiaVal&" + str(personal[1]) + "_individualFactor&" + str(social[0]) + "_socialFactor_median.csv"
filePath3_5= parentFile + "/graphData/" + str(no_dims[0]) + "_Dimensions&" + str(interia[0]) + "_InertiaVal&" + str(personal[2]) + "_individualFactor&" + str(social[0]) + "_socialFactor_median.csv"

filePath1_1= parentFile + "/graphData/" + str(no_dims[1]) + "_Dimensions&" + str(interia[0]) + "_InertiaVal&" + str(personal[0]) + "_individualFactor&" + str(social[0]) + "_socialFactor_median.csv"
filePath2_1= parentFile + "/graphData/" + str(no_dims[1]) + "_Dimensions&" + str(interia[0]) + "_InertiaVal&" + str(personal[1]) + "_individualFactor&" + str(social[0]) + "_socialFactor_median.csv"
filePath3_1= parentFile + "/graphData/" + str(no_dims[1]) + "_Dimensions&" + str(interia[0]) + "_InertiaVal&" + str(personal[2]) + "_individualFactor&" + str(social[0]) + "_socialFactor_median.csv"

data1_T = genfromtxt(filePath1_5, delimiter=',')
data2_T = genfromtxt(filePath2_5, delimiter=',')
data3_T = genfromtxt(filePath3_5, delimiter=',')

data1_F = genfromtxt(filePath1_1, delimiter=',')
data2_F = genfromtxt(filePath2_1, delimiter=',')
data3_F = genfromtxt(filePath3_1, delimiter=',')

plot(data1_T, data2_T, data3_T, ["Mjera individualnog faktora = 0.5", "Mjera individualnog faktora = 1.0", "Mjera individualnog faktora = 1.5"], "Ovisnost o mjeri individualnog faktora za dimenziju 5")

plot(data1_F, data2_F, data3_F, ["Mjera individualnog faktora = 0.5", "Mjera individualnog faktora = 1.0", "Mjera individualnog faktora = 1.5"], "Ovisnost o mjeri individualnog faktora za dimenziju 10")

print("Saved the plotted files")


#Changing social value

filePath1_5= parentFile + "/graphData/" + str(no_dims[0]) + "_Dimensions&" + str(interia[0]) + "_InertiaVal&" + str(personal[0]) + "_individualFactor&" + str(social[0]) + "_socialFactor_median.csv"
filePath2_5= parentFile + "/graphData/" + str(no_dims[0]) + "_Dimensions&" + str(interia[0]) + "_InertiaVal&" + str(personal[0]) + "_individualFactor&" + str(social[1]) + "_socialFactor_median.csv"
filePath3_5= parentFile + "/graphData/" + str(no_dims[0]) + "_Dimensions&" + str(interia[0]) + "_InertiaVal&" + str(personal[0]) + "_individualFactor&" + str(social[2]) + "_socialFactor_median.csv"

filePath1_1= parentFile + "/graphData/" + str(no_dims[1]) + "_Dimensions&" + str(interia[0]) + "_InertiaVal&" + str(personal[0]) + "_individualFactor&" + str(social[0]) + "_socialFactor_median.csv"
filePath2_1= parentFile + "/graphData/" + str(no_dims[1]) + "_Dimensions&" + str(interia[0]) + "_InertiaVal&" + str(personal[0]) + "_individualFactor&" + str(social[1]) + "_socialFactor_median.csv"
filePath3_1= parentFile + "/graphData/" + str(no_dims[1]) + "_Dimensions&" + str(interia[0]) + "_InertiaVal&" + str(personal[0]) + "_individualFactor&" + str(social[2]) + "_socialFactor_median.csv"

data1_T = genfromtxt(filePath1_5, delimiter=',')
data2_T = genfromtxt(filePath2_5, delimiter=',')
data3_T = genfromtxt(filePath3_5, delimiter=',')

data1_F = genfromtxt(filePath1_1, delimiter=',')
data2_F = genfromtxt(filePath2_1, delimiter=',')
data3_F = genfromtxt(filePath3_1, delimiter=',')

plot(data1_T, data2_T, data3_T, ["Mjera socijalnog faktora = 0.5", "Mjera socijalnog faktora = 1.0", "Mjera socijalnog faktora = 1.5"], "Ovisnost o mjeri socijalnog faktora za dimenziju 5")

plot(data1_F, data2_F, data3_F, ["Mjera socijalnog faktora = 0.5", "Mjera socijalnog faktora = 1.0", "Mjera socijalnog faktora = 1.5"], "Ovisnost o mjeri socijalnog faktora za dimenziju 10")

print("Saved the plotted files")
