# -*- coding: utf-8 -*-

import sys
import os
import random
import math

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
NO_QUEENS = 8   #table size and also the size of the cromosome
NGEN = 5000 #number of generations
POP_SIZE = 100  #population size
MUTPB = 0.2 #probability for mutating an individual
NELT = 4    #number of elite individuals
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

def generateQueenImage(individual):
        #Find out quuens who are in baad position
        bad = [False]* NO_QUEENS
        for i in range(NO_QUEENS-1):
            for j in range(i+1, NO_QUEENS):
                g1 = individual[i]
                g2 = individual[j]
                if (g1 == g2) or (j - i == math.fabs(g1 - g2)):
                    bad[i] = True
                    bad[j] = True

        #Create a transparent image
        img = QImage(1000, 1000, QImage.Format_ARGB32)
        img.fill(Qt.transparent)
        
        #Create a painter
        painter = QPainter(img)
        
        #cell size
        cell_size = 1000 / NO_QUEENS
        
        #Draw queens to the world image     
        x_offset = 0
        for i in range(NO_QUEENS):
            painter.drawImage(QRect(x_offset, individual[i]*cell_size, cell_size, cell_size), queen_img)
            x_offset += cell_size
        
        #Draw invalid error signs    
        x_offset = 0
        for i in range(NO_QUEENS):
            if bad[i]:
                painter.drawImage(QRect(x_offset, individual[i]*cell_size, cell_size, cell_size), error_img)
            x_offset += cell_size

        #Finish painter
        painter.end()
        
        #Return finished image
        return img
        

class MyQFrame(QtWidgets.QFrame):
    def paintEvent(self, event):
        painterWorld = QPainter(self)
        painterWorld.drawPixmap(self.rect(), self.img)
        painterWorld.end()

class Ui_MainWindow(QtWidgets.QMainWindow):
    def setupUi(self):
        self.setObjectName("MainWindow")
        self.resize(600, 830)
        self.setWindowTitle("GA - Queens")
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.frameWorld = MyQFrame(self.centralwidget)
        self.frameWorld.img = QPixmap(1000,1000)
        self.frameWorld.setGeometry(QtCore.QRect(10, 10, 400, 400))
        self.frameWorld.setFrameShape(QtWidgets.QFrame.Box)
        self.frameWorld.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frameWorld.setObjectName("frameWorld")
        self.frameChart = QChartView(self.centralwidget)
        self.frameChart.setGeometry(QtCore.QRect(10, 420, 400, 400))
        self.frameChart.setFrameShape(QtWidgets.QFrame.Box)
        self.frameChart.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frameChart.setRenderHint(QPainter.Antialiasing)
        self.frameChart.setObjectName("frameChart")
        self.gaParams = QtWidgets.QGroupBox(self.centralwidget)
        self.gaParams.setGeometry(QtCore.QRect(430, 10, 161, 171))
        self.gaParams.setObjectName("gaParams")
        self.gaParams.setTitle("GA parameters")
        self.label1 = QtWidgets.QLabel(self.gaParams)
        self.label1.setGeometry(QtCore.QRect(10, 20, 61, 16))
        self.label1.setObjectName("label1")
        self.label1.setText("Population:")
        self.label2 = QtWidgets.QLabel(self.gaParams)
        self.label2.setGeometry(QtCore.QRect(10, 50, 47, 16))
        self.label2.setObjectName("label2")
        self.label2.setText("Mutation:")
        self.label3 = QtWidgets.QLabel(self.gaParams)
        self.label3.setGeometry(QtCore.QRect(10, 80, 81, 16))
        self.label3.setObjectName("label3")
        self.label3.setText("Elite members:")
        self.label4 = QtWidgets.QLabel(self.gaParams)
        self.label4.setGeometry(QtCore.QRect(10, 110, 91, 16))
        self.label4.setObjectName("label4")
        self.label4.setText("No. generations:")
        self.cbxPermutation = QtWidgets.QCheckBox(self.gaParams)
        self.cbxPermutation.setGeometry(QtCore.QRect(35, 140, 91, 17))
        self.cbxPermutation.setObjectName("cbxPermutation")
        self.cbxPermutation.setText("Permutation")
        self.tbxPopulation = QtWidgets.QLineEdit(self.gaParams)
        self.tbxPopulation.setGeometry(QtCore.QRect(100, 20, 51, 20))
        self.tbxPopulation.setObjectName("tbxPopulation")
        self.tbxMutation = QtWidgets.QLineEdit(self.gaParams)
        self.tbxMutation.setGeometry(QtCore.QRect(100, 50, 51, 20))
        self.tbxMutation.setObjectName("tbxMutation")
        self.tbxElite = QtWidgets.QLineEdit(self.gaParams)
        self.tbxElite.setGeometry(QtCore.QRect(100, 80, 51, 20))
        self.tbxElite.setObjectName("tbxElite")
        self.tbxGenerations = QtWidgets.QLineEdit(self.gaParams)
        self.tbxGenerations.setGeometry(QtCore.QRect(100, 110, 51, 20))
        self.tbxGenerations.setObjectName("tbxGenerations")
        self.label5 = QtWidgets.QLabel(self.centralwidget)
        self.label5.setGeometry(QtCore.QRect(440, 190, 61, 16))
        self.label5.setObjectName("label5")
        self.label5.setText("No. queens:")
        self.tbxNoQueens = QtWidgets.QLineEdit(self.centralwidget)
        self.tbxNoQueens.setGeometry(QtCore.QRect(510, 190, 51, 20))
        self.tbxNoQueens.setObjectName("tbxNoQueens")
        self.cbxNoVis = QtWidgets.QCheckBox(self.centralwidget)
        self.cbxNoVis.setGeometry(QtCore.QRect(420, 215, 170, 17))
        self.cbxNoVis.setObjectName("cbxNoVis")
        self.cbxNoVis.setText("No visualization per generation")
        self.btnStart = QtWidgets.QPushButton(self.centralwidget)
        self.btnStart.setGeometry(QtCore.QRect(430, 250, 75, 23))
        self.btnStart.setObjectName("btnStart")
        self.btnStart.setText("Start")
        self.btnStop = QtWidgets.QPushButton(self.centralwidget)
        self.btnStop.setEnabled(False)
        self.btnStop.setGeometry(QtCore.QRect(510, 250, 75, 23))
        self.btnStop.setObjectName("btnStop")
        self.btnStop.setText("Stop")
        self.btnSaveWorld = QtWidgets.QPushButton(self.centralwidget)
        self.btnSaveWorld.setGeometry(QtCore.QRect(430, 370, 121, 41))
        self.btnSaveWorld.setObjectName("btnSaveWorld")
        self.btnSaveWorld.setText("Save world as image")
        self.btnSaveChart = QtWidgets.QPushButton(self.centralwidget)
        self.btnSaveChart.setGeometry(QtCore.QRect(430, 730, 121, 41))
        self.btnSaveChart.setObjectName("btnSaveChart")
        self.btnSaveChart.setText("Save chart as image")
        self.btnSaveChartSeries = QtWidgets.QPushButton(self.centralwidget)
        self.btnSaveChartSeries.setGeometry(QtCore.QRect(430, 780, 121, 41))
        self.btnSaveChartSeries.setObjectName("btnSaveChartSeries")
        self.btnSaveChartSeries.setText("Save chart as series")
        self.setCentralWidget(self.centralwidget)
        QtCore.QMetaObject.connectSlotsByName(self)
        
        #Connect events
        self.btnStart.clicked.connect(self.btnStart_Click)
        self.btnStop.clicked.connect(self.btnStop_Click)
        self.btnSaveWorld.clicked.connect(self.btnSaveWorld_Click)
        self.btnSaveChart.clicked.connect(self.btnSaveChart_CLick)
        self.btnSaveChartSeries.clicked.connect(self.btnSaveChartSeries_Click)
        
        #Set default GA variables
        self.tbxNoQueens.insert(str(NO_QUEENS))
        self.tbxGenerations.insert(str(NGEN))
        self.tbxPopulation.insert(str(POP_SIZE))
        self.tbxMutation.insert(str(MUTPB))
        self.tbxElite.insert(str(NELT))
        
        self.new_image = QPixmap(1000,1000)
        
    def btnStart_Click(self):
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
        NO_QUEENS = int(self.tbxNoQueens.text())
        NGEN = int(self.tbxGenerations.text())
        POP_SIZE = int(self.tbxPopulation.text())
        MUTPB = float(self.tbxMutation.text())
        NELT = int(self.tbxElite.text())
        
        #Painting chess table
        self.img = QPixmap(1000,1000)
        self.img.fill()
        painter = QPainter(self.img)
        painter.setPen(QPen(Qt.black,  10, Qt.SolidLine))
        width = 1000 / NO_QUEENS
        cur_width = 0
        for i in range(NO_QUEENS + 1): #+1 in order to draw the last line as well
            painter.drawLine(cur_width, 0, cur_width, 1000)
            painter.drawLine(0, cur_width, 1000, cur_width)
            cur_width += width
        painter.end()
        self.frameWorld.img = self.img
        #Redrawing frames
        self.frameWorld.repaint()
        app.processEvents()
        
        ####Initialize deap GA objects####
        
        #Make creator that minimize. If it would be 1.0 instead od -1.0 than it would be maxmize
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

        #Create an individual (a blueprint for cromosomes) as a list with a specified fitness type
        creator.create("Individual", list, fitness=creator.FitnessMin)

        #Create base toolbox for finishing creation of a individual (cromosome)
        self.toolbox = base.Toolbox()
        
        #Define what type of data (number, gene) will it be in the cromosome
        if self.cbxPermutation.isChecked():
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
        if self.cbxPermutation.isChecked():
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
        
        #Disable start and enable stop
        self.btnStart.setEnabled(False)
        self.btnStop.setEnabled(True)
        self.gaParams.setEnabled(False)
        self.tbxNoQueens.setEnabled(False)
        self.cbxNoVis.setEnabled(False)
        
        #Start evolution
        self.evolve()
        
    
    def btnStop_Click(self):
        global stop_evolution
        stop_evolution = True
        #Disable stop and enable start
        self.btnStop.setEnabled(False)
        self.btnStart.setEnabled(True)
        self.gaParams.setEnabled(True)
        self.tbxNoQueens.setEnabled(True)
        self.cbxNoVis.setEnabled(True)
    
    #Function for GA evolution
    def evolve(self):
        global q_min_series
        global q_max_series
        global q_avg_series
        
        # Variable for keeping track of the number of generations
        curr_g = 0
        
        # Begin the evolution till goal is reached or max number of generation is reached
        while min(self.fits) != 0 and curr_g < NGEN:
            #Check if evolution and thread need to stop
            if stop_evolution:
                break #Break the evolution loop
            
            # A new generation
            curr_g = curr_g + 1
            print("-- Generation %i --" % curr_g)
            
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
            
            print("  Evaluated %i individuals" % len(invalid_ind))
            
            #Replace population with offspring
            self.pop[:] = offspring
            
            # Gather all the fitnesses in one list and print the stats
            self.fits = [ind.fitness.values[0] for ind in self.pop]
            
            length = len(self.pop)
            mean = sum(self.fits) / length
            sum2 = sum(x*x for x in self.fits)
            std = abs(sum2 / length - mean**2)**0.5
            
            q_min_series.append(curr_g, min(self.fits))
            q_max_series.append(curr_g, max(self.fits))
            q_avg_series.append(curr_g, mean)
                      
            print("  Min %s" % q_min_series.at(q_min_series.count()-1).y())
            print("  Max %s" % q_max_series.at(q_max_series.count()-1).y())
            print("  Avg %s" % mean)
            print("  Std %s" % std)
            
            if self.cbxNoVis.isChecked():
                app.processEvents()
            else:
                #Draw queen positions of best individual on a image
                best_ind = tools.selBest(self.pop, 1)[0]
                self.updateWorldFrame(generateQueenImage(best_ind))
                
                self.chart = QChart()
                self.chart.addSeries(q_min_series)
                self.chart.addSeries(q_max_series)
                self.chart.addSeries(q_avg_series)
                self.chart.setTitle("Fitness value over time")
                self.chart.setAnimationOptions(QChart.NoAnimation)
                self.chart.createDefaultAxes()
                self.frameChart.setChart(self.chart)
                   
        #Printing best individual
        best_ind = tools.selBest(self.pop, 1)[0]
        print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
        
        #Visulaize final solution
        if self.cbxNoVis.isChecked():
            #Draw queen positions of best individual on a image
            best_ind = tools.selBest(self.pop, 1)[0]
            self.updateWorldFrame(generateQueenImage(best_ind))
            
            self.chart = QChart()
            self.chart.addSeries(q_min_series)
            self.chart.addSeries(q_max_series)
            self.chart.addSeries(q_avg_series)
            self.chart.setTitle("Fitness value over time")
            self.chart.setAnimationOptions(QChart.NoAnimation)
            self.chart.createDefaultAxes()
            self.frameChart.setChart(self.chart)
        
        #Disable stop and enable start
        self.btnStop.setEnabled(False)
        self.btnStart.setEnabled(True)
        self.gaParams.setEnabled(True)
        self.tbxNoQueens.setEnabled(True)
        self.cbxNoVis.setEnabled(True)
        
    def updateWorldFrame(self, queens_img):
        #new_image = QPixmap(1000,1000)
        self.new_image.fill() #White color is default
        painter = QPainter(self.new_image)
        #First draw the table
        painter.drawPixmap(self.new_image.rect(), self.img)
        #Then draw the queens
        painter.drawImage(self.new_image.rect(), queens_img)
        painter.end()
        #Set new image to the frame
        self.frameWorld.img = self.new_image
        #Redrawing frames
        self.frameWorld.repaint()
        self.frameChart.repaint()
        app.processEvents()
    
    def btnSaveWorld_Click(self):
        filename, _ = QFileDialog.getSaveFileName(None,"Save world as a image","","Image Files (*.png)")
        self.frameWorld.img.save(filename, "PNG");
        print ("World image saved to: ", filename)
    
    def btnSaveChart_CLick(self):
        p = self.frameChart.grab()
        filename, _ = QFileDialog.getSaveFileName(None,"Save series chart as a image","","Image Files (*.png)")
        p.save(filename, "PNG");
        print ("Chart series image saved to: ", filename)
    
    def btnSaveChartSeries_Click(self):
        global q_min_series
        global q_max_series
        global q_avg_series
        filename, _ = QFileDialog.getSaveFileName(None,"Save series to text file","","Text Files (*.txt, *.csv)")
        with open(filename, 'w') as dat:
            for i in range(q_min_series.count()):
                dat.write('%f,%f,%f\n' % (q_min_series.at(i).y(), q_avg_series.at(i).y(), q_max_series.at(i).y()))
        print ("Chart series saved to: ", filename)
    


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    # ui.setupUi()
    # ui.show()
    # sys.exit(app.exec_())
    
