# -*- coding: utf-8 -*-

import sys
import os
import random
import math
import pandas

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPainter, QBrush, QPen, QPixmap, QImage
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtChart import QLineSeries, QChart, QValueAxis, QChartView
from PyQt5.QtWidgets import QFileDialog

from deap import base, creator, tools

#Change current working directory to this script path
import pathlib
pathlib.Path(__file__).parent.absolute()
os.chdir(pathlib.Path(__file__).parent.absolute())

####Global GA parameters####
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
#########################

####Other global variables####
stop_evolution = False
q_min_series = QLineSeries()
q_min_series.setName("MIN")
q_max_series = QLineSeries()
q_max_series.setName("MAX")
q_avg_series = QLineSeries()
q_avg_series.setName("AVG")
queen_img = QImage("queen.png")
error_img = QImage("error.png")
##############################

#Define evaluation (fitness) function for individual (cromosome)
def evaluateInd(individual):
    fit_val = 0 #starting fitness is 0
    for i in range(NO_QUEENS-1):
        for j in range(i+1, NO_QUEENS):
            g1 = individual[i]
            g2 = individual[j]
            if (g1 == g2) or (j - i == math.fabs(g1 - g2)):
                fit_val += 1
    return fit_val,#returning must be a tuple becos of posibility of optimization via multiple goal values (objectives)
    
        
class Ui_MainWindow(QtWidgets.QMainWindow):
        
    def btnStart_Click(self, numQueens, numGeneration, populationSize, mutationProb, numElites, permutation):
        #Set global variables
        global stop_evolution
        global q_min_series
        global q_max_series
        global q_avg_series
        stop_evolution = False    
        q_min_series.clear()      
        q_max_series.clear()    
        q_avg_series.clear()
        
        #Set global variables from information on UI
        global NO_QUEENS
        global NGEN
        global POP_SIZE 
        global MUTPB
        global NELT
        NO_QUEENS = numQueens
        NGEN = numGeneration
        POP_SIZE = populationSize
        MUTPB = mutationProb
        NELT = numElites
        
        ####Initialize deap GA objects####
        
        #Make creator that minimize. If it would be 1.0 instead od -1.0 than it would be maxmize
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

        #Create an individual (a blueprint for cromosomes) as a list with a specified fitness type
        creator.create("Individual", list, fitness=creator.FitnessMin)

        #Create base toolbox for finishing creation of a individual (cromosome)
        self.toolbox = base.Toolbox()
        
        #Define what type of data (number, gene) will it be in the cromosome
        if permutation:
            #Permutation coding
            self.toolbox.register("indices", random.sample, range(NO_QUEENS), NO_QUEENS)
            #initIterate requires that the generator of genes (such as random.sample) generates an iterable (a list) variable
            self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.indices)
        else:
            #Standard coding
            self.toolbox.register("attr_int", random.randint, 0, NO_QUEENS - 1) #number in cromosome is from 0 till IND_SIZE - 1
            #Initialization procedure (initRepeat) for the cromosome. For the individual to be completed we need to run initRepeat for the amaout of genes the cromosome includes
            self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_int, n=NO_QUEENS)

        #Create a population of individuals (cromosomes). The population is then created by toolbox.population(n=300) where 'n' is the number of cromosomes in population
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        #Register evaluation function
        self.toolbox.register("evaluate", evaluateInd)

        #Register what genetic operators to use
        if permutation:
            #Permutation coding
            self.toolbox.register("mate", tools.cxUniformPartialyMatched, indpb=0.2)#Use uniform recombination for permutation coding
            self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
        else:
            #Standard coding
            self.toolbox.register("mate", tools.cxTwoPoint)#Use two point recombination
            self.toolbox.register("mutate", tools.mutUniformInt, low=0, up=NO_QUEENS-1, indpb=0.2)   #20% that the gene will change

        self.toolbox.register("select", tools.selTournament, tournsize=3)    #Use tournament selection
        
        ##################################
        
        #Generate initial poplation. Will be a member variable so we can easely pass everything to new thread
        self.pop = self.toolbox.population(n=POP_SIZE)
    
        #Evaluate initial population, we map() the evaluation function to every individual and then assign their respective fitness, map runs evaluate function for each individual in pop
        fitnesses = list(map(self.toolbox.evaluate, self.pop))
        for ind, fit in zip(self.pop, fitnesses):
            ind.fitness.values = fit    #Assign calcualted fitness value to individuals
        
        #Extracting all the fitnesses of all individuals in a population so we can monitor and evovlve the algorithm until it reaches 0 or max number of generation is reached
        self.fits = [ind.fitness.values[0] for ind in self.pop]
        
        #Start evolution
        self.evolve(self)

    #Function for GA evolution
    def evolve(self):
        global q_min_series
        global q_max_series
        global q_avg_series
        
        # Variable for keeping track of the number of generations
        self.curr_g = 0
        
        # Begin the evolution till goal is reached or max number of generation is reached
        while min(self.fits) != 0 and self.curr_g < NGEN:
            #Check if evolution and thread need to stop
            if stop_evolution:
                break #Break the evolution loop
            
            # A new generation
            self.curr_g = self.curr_g + 1
            if self.curr_g % 5000 == 0:
                print("-- NUM_QUEENS = %i --\n" % NO_QUEENS)
                print("-- Generation %i -- " % self.curr_g)
            
            # Select the next generation individuals
            #Select POP_SIZE - NELT number of individuals. Since recombination is between neigbours, not two naighbours should be the clone of the same individual
            offspring = []
            offspring.append(self.toolbox.select(self.pop, 1)[0])    #add first selected individual
            for i in range(POP_SIZE - NELT - 1):    # -1 because the first seleceted individual is already added
                while True:
                    new_o = self.toolbox.select(self.pop, 1)[0]
                    if new_o != offspring[len(offspring) - 1]:   #if it is different than the last inserted then add to offspring and break
                        offspring.append(new_o)
                        break
            
            # Clone the selected individuals because all of the changes are inplace
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Apply crossover on the selected offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                self.toolbox.mate(child1, child2)    #inplace recombination
                #Invalidate new children fitness values
                del child1.fitness.values
                del child2.fitness.values
    
            #Apply mutation on the offspring
            for mutant in offspring:
                if random.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            #Add elite individuals #Is clonning needed?
            offspring.extend(list(map(self.toolbox.clone, tools.selBest(self.pop, NELT))))         
                    
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            if self.curr_g % 5000 == 0:
                print("  Evaluated %i individuals" % len(invalid_ind))
            
            #Replace population with offspring
            self.pop[:] = offspring
            
            # Gather all the fitnesses in one list and print the stats
            self.fits = [ind.fitness.values[0] for ind in self.pop]
            
            length = len(self.pop)
            mean = sum(self.fits) / length
            sum2 = sum(x*x for x in self.fits)
            std = abs(sum2 / length - mean**2)**0.5
            
            q_min_series.append(self.curr_g, min(self.fits))
            q_max_series.append(self.curr_g, max(self.fits))
            q_avg_series.append(self.curr_g, mean)
                      
            if self.curr_g % 5000 == 0:        
                print("  Min %s" % q_min_series.at(q_min_series.count()-1).y())
                print("  Max %s" % q_max_series.at(q_max_series.count()-1).y())
                print("  Avg %s" % mean)
                print("  Std %s" % std)
            
                   
        #Printing best individual
        best_ind = tools.selBest(self.pop, 1)[0]
        print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

    def autoRun(self):

        global q_min_series

        numQueens = [NUM_QUEENS12, NUM_QUEENS24, NUM_QUEENS48]
        populationSize = [POPULATION_SIZE50, POPULATION_SIZE100, POPULATION_SIZE200]
        numOfElites = [NUM_ELITES4,NUM_ELITES8, NUM_ELITES16]
        mutationRate = [MUTATION_PROB4, MUTATION_PROB8, MUTATION_PROB16]
        permutations = [True, False]
        numOfRuns = 0
        successCount = 0
        lowestGenCnt = 50000
        genAvg = 0 

        folderExists = os.path.exists("stats/")

        if not folderExists:
            os.makedirs("stats/")
            print("Created stats directory!")

        folderExists = os.path.exists("graphData/")
        if not folderExists:
            os.makedirs("graphData/")
            print("Created graph data directory!")

        lastIterationExists = os.path.exists("lastIteration.txt")
        if lastIterationExists:
            f = open("lastIteration.txt")
            text=f.read()
            text=text.split(",")
            startQ=int(text[0])
            startPER=int(text[1])
            starti=int(text[2])
        else:
            startQ=0
            startPER=0
            starti=0
        startQ=2
        startPER=0
        starti=1
        print("Latest iteration:")
        print("Q=" , startQ)
        print("PER=" , startPER)
        print("i=" , starti)
        
        for Q in range(startQ,3):
            for PER in range(startPER,2):
                for i in range(starti,7):
                    startQ=0
                    startPER=0
                    starti=0
                    #default value:
                    POP=0
                    M=0
                    E=0
                    if i==1:
                        POP=1
                    if i==2:
                        POP=2
                    if i==3:
                        M=1
                    if i==4:
                        M=2
                    if i==5:
                        E=1
                    if i==6:
                        E=2
                    median = []
                    #while(successCount<5 and numOfRuns<=30): #max number of tries or 5 succesful ones
                    self.btnStart_Click(self,
                                        numQueens=numQueens[Q],
                                        numElites=numOfElites[E],
                                        mutationProb=mutationRate[M],
                                        populationSize=populationSize[POP],
                                        numGeneration=100000,
                                        permutation=permutations[PER])
                        #if ((min(self.fits) == 0) or numOfRuns-successCount>=25):
                    successCount += 1
                    genAvg += self.curr_g
                    values = []
                    for g in range(q_min_series.count()):
                        values.append(q_min_series.at(g).y())
                    median.append(values)
                    numOfRuns += 1
                    if self.curr_g < lowestGenCnt:
                        lowestGenCnt = self.curr_g
                    print("successCount: ", successCount, "\nnumOfRuns: ", numOfRuns)
                    values = []

                    median.sort(key=medianLen)
                    graphDataFname = "graphData/Queens=" + str(numQueens[Q]) + "_popSize=" + str(populationSize[POP]) + "_numElites=" + str(numOfElites[E]) + "_mutationRate=" + str(mutationRate[M]) + "_permutations=" + str(permutations[PER]) + ".csv"
                    #dataFrame = pandas.DataFrame(median[2])
                    dataFrame = pandas.DataFrame(median)
                    dataFrame.to_csv(graphDataFname, index=True, header=False)
                    print(dataFrame.shape)
                    print(dataFrame)
                    print("Median stats written!")


                    statsfName = "stats/" + str(numQueens[Q]) + "_queens_" + str(populationSize[POP]) + "_popSize_" + str(numOfElites[E]) + "_elites_" + str(mutationRate[M]) + "_mutationRate_" + str(permutations[PER]) + "_permutations.txt"
                    stats = str(numQueens[Q]) + " Queens, " + str(populationSize[POP]) + " Population Size, " + str(numOfElites[E]) + " Number of Elites, " + str(mutationRate[M]) + " Mutation Rate & permutation is " + str(permutations[PER]) + " stats: \nlowestGenCount=" + str(lowestGenCnt) + "\nAvgGenCount="  + str(genAvg/5) + "\nnumber of runs=" + str(numOfRuns) #+"\nGenerations found:%i,%i,%i,%i,%i" %(len(median[0]),len(median[1]),len(median[2]),len(median[3]),len(median[4])) 
                    with open(statsfName, 'w') as dat:
                        dat.write(stats)
                    print("Stats for %i queens, population size of %i, %i number of elites and %f percent mutation rate (permutation is set to %r) run written in %s file" % (numQueens[Q], populationSize[POP], numOfElites[E], mutationRate[M], permutations[PER], statsfName))

                    successCount = 0
                    numOfRuns = 0
                    lowestGenCnt = 50000
                    genAvg = 0


                    lastIterationFname="lastIteration.txt"
                    lastIteration=str(Q) + "," + str(PER) + "," + str(i)
                    with open(lastIterationFname, "w") as dat:
                        dat.write(lastIteration)

def medianLen(values):
    return len(values)       

if __name__ == "__main__":
    app = Ui_MainWindow
    app.autoRun(app)
    
