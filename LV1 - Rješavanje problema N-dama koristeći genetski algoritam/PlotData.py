from matplotlib import pyplot as plt
import csv
import pandas as pd
from numpy import genfromtxt
import numpy as np

NUM_QUEENS12 = 12
NUM_QUEENS24 = 24
NUM_QUEENS48 = 48

POPULATION_SIZE50 = 50
POPULATION_SIZE100 = 100
POPULATION_SIZE200 = 200

MUTATION_PROB4 = 0.04
MUTATION_PROB8 = 0.08
MUTATION_PROB16 = 0.16

NUM_ELITES4 = 4
NUM_ELITES8 = 8
NUM_ELITES16 = 16

numQueens = [NUM_QUEENS12, NUM_QUEENS24, NUM_QUEENS48]
populationSize = [POPULATION_SIZE50, POPULATION_SIZE100, POPULATION_SIZE200]
numOfElites = [NUM_ELITES4,NUM_ELITES8, NUM_ELITES16]
mutationRate = [MUTATION_PROB4, MUTATION_PROB8, MUTATION_PROB16]
permutations = [True, False]

filePath1_T="graphData/Queens=" + str(numQueens[2]) + "_popSize=" + str(populationSize[0]) + "_numElites=" + str(numOfElites[0]) + "_mutationRate=" + str(mutationRate[0]) + "_permutations=" + str(permutations[0]) + ".csv"
filePath2_T="graphData/Queens=" + str(numQueens[2]) + "_popSize=" + str(populationSize[1]) + "_numElites=" + str(numOfElites[0]) + "_mutationRate=" + str(mutationRate[0]) + "_permutations=" + str(permutations[0]) + ".csv"
filePath3_T="graphData/Queens=" + str(numQueens[2]) + "_popSize=" + str(populationSize[2]) + "_numElites=" + str(numOfElites[0]) + "_mutationRate=" + str(mutationRate[0]) + "_permutations=" + str(permutations[0]) + ".csv"

filePath1_F="graphData/Queens=" + str(numQueens[2]) + "_popSize=" + str(populationSize[0]) + "_numElites=" + str(numOfElites[0]) + "_mutationRate=" + str(mutationRate[0]) + "_permutations=" + str(permutations[1]) + ".csv"
filePath2_F="graphData/Queens=" + str(numQueens[2]) + "_popSize=" + str(populationSize[1]) + "_numElites=" + str(numOfElites[0]) + "_mutationRate=" + str(mutationRate[0]) + "_permutations=" + str(permutations[1]) + ".csv"
filePath3_F="graphData/Queens=" + str(numQueens[2]) + "_popSize=" + str(populationSize[2]) + "_numElites=" + str(numOfElites[0]) + "_mutationRate=" + str(mutationRate[0]) + "_permutations=" + str(permutations[1]) + ".csv"


data1_T = genfromtxt(filePath1_T, delimiter=',')
data2_T = genfromtxt(filePath2_T, delimiter=',')
data3_T = genfromtxt(filePath3_T, delimiter=',')

data1_F = genfromtxt(filePath1_F, delimiter=',')
data2_F = genfromtxt(filePath2_F, delimiter=',')
data3_F = genfromtxt(filePath3_F, delimiter=',')

# f=open(filePath3_F, "a")
# for i in range(100001-len(data3_F)):
#     f.write("1.0,")

print(len(data1_T))
print(len(data2_T))
print(len(data3_T))

print(len(data1_F))
print(len(data2_F))
print(len(data3_F))