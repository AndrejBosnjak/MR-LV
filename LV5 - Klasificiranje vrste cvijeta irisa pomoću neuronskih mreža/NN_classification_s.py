# -*- coding: utf-8 -*-

import sys
import os
import random
import math

import networkx as nx

from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import unique_labels

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPainter, QBrush, QPen, QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtWidgets import QFileDialog, QWidget, QVBoxLayout

# Make sure that we are using QT5
import matplotlib
matplotlib.use('Qt5Agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import numpy as np

#Change current working directory to this script path
import pathlib
pathlib.Path(__file__).parent.absolute()
os.chdir(pathlib.Path(__file__).parent.absolute())

####Global NN parameters######
N_PER_LAYER = (10,10) #Neurons per layer
ALPHA = 0.00001 #Alpha value, regularization term
ACTIVATION_F = 1 #Neuron activation function, 1 = "logistic"
SOLVER = 0 #Solver algorithm, 1 = "lbfgs"
MAX_ITER = 1000 #Max number of training iterations
GUI=False
##############################

####Other global variables####
#Load learn and test data
train_data = np.genfromtxt("iris_samples_train.csv", delimiter=',', dtype='float32', skip_header=1)
test_data = np.genfromtxt("iris_samples_test.csv", delimiter=',', dtype='float32', skip_header=1)
##############################

#Function for visualizing scikit-learn neural networks
def VisualizeNN(mpl_axes, mpl_figure, mlp, use_weighted_edges = True, show_neuron_labels = True, show_bias = False):
    #Determine number of layers. Needed for width of the graph
    no_layers = mlp.n_layers_

    g_height = 0
    layer_sizes = ()
    try:
        #Determine max number of neurons per layer. It will represent the height of the graph
        g_height = max(mlp.hidden_layer_sizes)
        #Determining number of neurons per layer
        layer_sizes = (mlp.coefs_[0].shape[0],) + mlp.hidden_layer_sizes + (mlp.n_outputs_,) #Imput layer + hidden layers + output layer
    except:
        #Determine max number of neurons per layer. It will represent the height of the graph
        g_height = mlp.hidden_layer_sizes
        #Determining number of neurons per layer
        layer_sizes = (mlp.coefs_[0].shape[0],) + (mlp.hidden_layer_sizes,) + (mlp.n_outputs_,) #Imput layer + hidden layers + output layer
    
    
    #Crating graph
    g = nx.Graph()
    
    #Adding neurons per layer
    curr_n = 0
    for l in range(no_layers):
        for n in range(layer_sizes[l]):
            g.add_node (curr_n, pos=(l * 3, (-1.0) * (g_height / 2.0 - layer_sizes[l] / 2.0 + n))) #(-1.0 * ...) in order to be inverted on Y axes, othervise (0,0) is lower left corner
            curr_n += 1
    
    #Adding edges
    curr_n = 0
    start_next_layer_idx = 0
    for l in range(no_layers-1):
        start_next_layer_idx += layer_sizes[l]
        for n in range(layer_sizes[l]):
            for n_next in range(layer_sizes[l+1]):
                g.add_edge(curr_n, start_next_layer_idx + n_next, weight=mlp.coefs_[l][n][n_next])
            curr_n += 1
    
    #Add bias nodes if requested
    if show_bias:
        #Adding bias nodes
        for l in range(1, no_layers): #Start from first hidden layer
            g.add_node ('b' + str(l), pos=(l * 3 + 1.5, 1.0)) 
        
        #Adding bias edges
        curr_n = layer_sizes[0]
        for l in range(1, no_layers): #Start from first hidden layer
            for n in range(layer_sizes[l]):
                g.add_edge(curr_n, 'b' + str(l), weight=mlp.intercepts_[l-1][n])
                curr_n += 1
                
    #Drawing
    pos=nx.get_node_attributes(g,'pos')
    if use_weighted_edges:
        weights = nx.get_edge_attributes(g, "weight")
        weights_v = list(weights.values())
        nx.draw(g, pos, ax = mpl_axes, edge_color=weights_v, edge_cmap=plt.cm.Spectral, with_labels=show_neuron_labels, font_weight='bold')
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Spectral, norm=plt.Normalize(vmin = min(weights_v), vmax=max(weights_v)))
        sm._A = []
        mpl_figure.colorbar(sm)
        
    else:
        nx.draw(g, pos, ax = mpl_axes, with_labels=show_neuron_labels, font_weight='bold')
        
#Function for computing and displaying confusion matrices
def PlotConfusionMatrix(mpl_axes, mpl_figure, y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    im = mpl_axes.imshow(cm, interpolation='nearest', cmap=cmap)
    mpl_axes.figure.colorbar(im, ax=mpl_axes)
    # We want to show all ticks...
    mpl_axes.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(mpl_axes.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            mpl_axes.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    mpl_figure.tight_layout()

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)

        FigureCanvas.__init__(self, self.fig)
        
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
    def Reset(self):
        self.fig.clf()
        self.axes = self.fig.add_subplot(111)
        

class Ui_MainWindow(QtWidgets.QMainWindow):
    def setupUi(self):
        self.setObjectName("MainWindow")
        self.resize(1010, 915)
        self.setWindowTitle("NN - Approximation")
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        
        self.lineChartWidget = QtWidgets.QWidget(self.centralwidget)
        self.lineChartWidget.setGeometry(QtCore.QRect(10, 10, 600, 450))
        
        self.lineChart = MplCanvas(self.lineChartWidget, 600, 450)
        self.lineChart.setObjectName("lineChart")
        
        self.lineChartWidgetlayout = QtWidgets.QVBoxLayout()
        self.lineChartWidgetlayout.addWidget(self.lineChart)
        self.lineChartWidgetlayout.addWidget(NavigationToolbar(self.lineChart, self.lineChartWidget))
        self.lineChartWidget.setLayout(self.lineChartWidgetlayout)
        
        self.NNChartWidget = QtWidgets.QWidget(self.centralwidget)
        self.NNChartWidget.setGeometry(QtCore.QRect(10, 470, 1000, 450))
        
        self.NNChart = MplCanvas(self.NNChartWidget, 1000, 450)
        self.NNChart.setObjectName("NNChart")
        
        self.NNChartWidgetlayout = QtWidgets.QVBoxLayout()
        self.NNChartWidgetlayout.addWidget(self.NNChart)
        self.NNChartWidgetlayout.addWidget(NavigationToolbar(self.NNChart, self.NNChartWidget))
        self.NNChartWidget.setLayout(self.NNChartWidgetlayout)
        
        self.nnParams = QtWidgets.QGroupBox(self.centralwidget)
        self.nnParams.setGeometry(QtCore.QRect(630, 10, 200, 165))
        self.nnParams.setObjectName("nnParams")
        self.nnParams.setTitle("NN parameters")
        
        self.label1 = QtWidgets.QLabel(self.nnParams)
        self.label1.setGeometry(QtCore.QRect(10, 20, 91, 16))
        self.label1.setObjectName("label1")
        self.label1.setText("Neurons per layer:")
        
        self.label2 = QtWidgets.QLabel(self.nnParams)
        self.label2.setGeometry(QtCore.QRect(10, 50, 91, 16))
        self.label2.setObjectName("label2")
        self.label2.setText("Alpha:")
        
        self.label3 = QtWidgets.QLabel(self.nnParams)
        self.label3.setGeometry(QtCore.QRect(10, 80, 91, 16))
        self.label3.setObjectName("label3")
        self.label3.setText("Activation function:")
        
        self.label4 = QtWidgets.QLabel(self.nnParams)
        self.label4.setGeometry(QtCore.QRect(10, 110, 91, 16))
        self.label4.setObjectName("label4")
        self.label4.setText("Solver:")
        
        self.label5 = QtWidgets.QLabel(self.nnParams)
        self.label5.setGeometry(QtCore.QRect(10, 140, 91, 16))
        self.label5.setObjectName("label5")
        self.label5.setText("Max iterations:")
        
        self.tbxNeuroPerLayer = QtWidgets.QLineEdit(self.nnParams)
        self.tbxNeuroPerLayer.setGeometry(QtCore.QRect(120, 20, 71, 20))
        self.tbxNeuroPerLayer.setObjectName("tbxNeuroPerLayer")
        
        self.tbxAlpha = QtWidgets.QLineEdit(self.nnParams)
        self.tbxAlpha.setGeometry(QtCore.QRect(120, 50, 71, 20))
        self.tbxAlpha.setObjectName("tbxAlpha")
        
        self.comboActivation = QtWidgets.QComboBox(self.nnParams)
        self.comboActivation.setGeometry(QtCore.QRect(120, 80, 71, 20))
        self.comboActivation.setObjectName("comboActivation")
        self.comboActivation.addItem("identity")
        self.comboActivation.addItem("logistic")
        self.comboActivation.addItem("tanh")
        self.comboActivation.addItem("relu")
        
        self.comboSolver = QtWidgets.QComboBox(self.nnParams)
        self.comboSolver.setGeometry(QtCore.QRect(120, 110, 71, 20))
        self.comboSolver.setObjectName("comboSolver")
        self.comboSolver.addItem("lbfgs")
        self.comboSolver.addItem("sgd")
        self.comboSolver.addItem("adam")
        
        self.tbxMaxIter = QtWidgets.QLineEdit(self.nnParams)
        self.tbxMaxIter.setGeometry(QtCore.QRect(120, 140, 71, 20))
        self.tbxMaxIter.setObjectName("tbxMaxIter")
        
        self.btnStart = QtWidgets.QPushButton(self.centralwidget)
        self.btnStart.setGeometry(QtCore.QRect(650, 200, 150, 50))
        self.btnStart.setObjectName("btnStart")
        self.btnStart.setText("Start")
        
        self.cbxShowNeuronLabels = QtWidgets.QCheckBox(self.centralwidget)
        self.cbxShowNeuronLabels.setGeometry(QtCore.QRect(300, 460, 100, 17))
        self.cbxShowNeuronLabels.setObjectName("cbxShowNeuronLabels")
        self.cbxShowNeuronLabels.setText("Show neuron labels")
        
        self.cbxShowWeights = QtWidgets.QCheckBox(self.centralwidget)
        self.cbxShowWeights.setGeometry(QtCore.QRect(500, 460, 100, 17))
        self.cbxShowWeights.setObjectName("cbxShowWeights")
        self.cbxShowWeights.setText("Show weights")

        self.cbxShowBias = QtWidgets.QCheckBox(self.centralwidget)
        self.cbxShowBias.setGeometry(QtCore.QRect(700, 460, 100, 17))
        self.cbxShowBias.setObjectName("cbxShowBias")
        self.cbxShowBias.setText("Show bias")
        
        self.setCentralWidget(self.centralwidget)
        QtCore.QMetaObject.connectSlotsByName(self)
        
        #Connect events
        self.btnStart.clicked.connect(self.btnStart_Click)
        
        #Set default GA variables
        self.tbxNeuroPerLayer.insert(str(N_PER_LAYER))
        self.tbxAlpha.insert(str(ALPHA))
        self.comboActivation.setCurrentIndex(ACTIVATION_F)
        self.comboSolver.setCurrentIndex(SOLVER)
        self.tbxMaxIter.insert(str(MAX_ITER))
        
    def btnStart_Click(self, nNeurons, nLayers, activationFunc, solverFunc):
    #def btnStart_Click(self):
        #Set global variables from information on UI
        global N_PER_LAYER
        global ALPHA
        global ACTIVATION_F
        global SOLVER
        global MAX_ITER
        if(GUI):
            N_PER_LAYER = eval(self.tbxNeuroPerLayer.text())
            ALPHA = float(self.tbxAlpha.text())
            ACTIVATION_F = self.comboActivation.currentIndex()
            SOLVER = self.comboSolver.currentIndex()
            MAX_ITER = int(self.tbxMaxIter.text())
        else:
            if(nLayers == 1 ):
                N_PER_LAYER = (nNeurons)
            elif(nLayers == 2):
                N_PER_LAYER = (nNeurons,nNeurons)
            elif(nLayers == 3):
                N_PER_LAYER = (nNeurons,nNeurons,nNeurons)
            ACTIVATION_F=activationFunc
            SOLVER=solverFunc

        a_functions=["identity", "logistic", "tanh", "relu"]
        solvers=['lbfgs', 'sgd', 'adam']
        
        #Create neural network
        #Ispravno instancirati odgovarajuci tip neurnonskih mreza, uzimajuci u obzir podatke zadane preko sucelja koje su pohranjene u iznad navedene globalne varijable
        mlp = MLPClassifier(hidden_layer_sizes=N_PER_LAYER, activation=a_functions[ACTIVATION_F], alpha=ALPHA, solver=solvers[SOLVER], max_iter=MAX_ITER)
        
        #Learn neural network
        mlp.fit(train_data[:,0:4], train_data[:,4])
        
        output = mlp.predict(test_data[:,0:4])
        
        #Round up the classification
        output = np.around(output).astype('int')
        if (GUI):
            #Draw confusion matrix
            self.lineChart.Reset()
            #Name of the classes, could also be strings
            class_names = np.array([0,1,2])
            PlotConfusionMatrix(self.lineChart.axes, self.lineChart.figure, test_data[:,4].astype('int'), output, classes=class_names, normalize=True)
            self.lineChart.draw()
        else:
            # Define the labels for the confusion matrix
            labels = ['0', '1', '2']

            # Define the confusion matrix as a 2D numpy array
            cm = confusion_matrix(test_data[:,4].astype('int'),output)

            # Normalize confusion matrix
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    

            # Create the figure and axis objects
            fig, ax = plt.subplots()

            # Create a heatmap of the confusion matrix
            im = ax.imshow(cm, cmap='Blues')

            # Add colorbar
            cbar = ax.figure.colorbar(im, ax=ax)

            # Show all ticks and label them with the respective list entries
            ax.set_xticks(np.arange(len(labels)))
            ax.set_yticks(np.arange(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)

            # Rotate the tick labels and set their alignment
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                    rotation_mode="anchor")

            # Loop over the data and add annotations to the heatmap
            for i in range(len(labels)):
                for j in range(len(labels)):
                    text = ax.text(j, i, cm[i, j],
                                ha="center", va="center", color="black")

            # Set the title and axis labels
            ax.set_title("Confusion matrix for {}nNeurons_{}nLayers_{}_{}".format(nNeurons, nLayers, a_functions[ACTIVATION_F], solvers[SOLVER]))
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")

            # Show the plot
            plt.savefig("graphs/{}nNeurons_{}nLayers_{}_{}".format(nNeurons, nLayers, a_functions[ACTIVATION_F], solvers[SOLVER]))
            
              
        #Draw NN chart
        if (GUI):
            self.NNChart.Reset()
            VisualizeNN(self.NNChart.axes, self.NNChart.figure, mlp, use_weighted_edges=self.cbxShowWeights.isChecked(), show_neuron_labels=self.cbxShowNeuronLabels.isChecked(), show_bias=self.cbxShowBias.isChecked())
            plt.close() #Close anomalous popping figure?!?
            self.NNChart.draw()
        
        return accuracy_score(test_data[:,4].astype('int'), output)
        

    def autorun(self):
        nNeurons_ = [5, 10, 15]
        nLayers_ = [1,2,3]
        activationFunc_ = [0, 1, 2, 3]
        solverFunc_ = [0, 1, 2]

        a_functions=["identity", "logistic", "tanh", "relu"]
        solvers=['lbfgs', 'sgd', 'adam']

        # if not os.path.exists("graphs/"):
        #     os.makedirs("graphs/")
        # if not os.path.exists("AccuracyValues/"):
        #     os.makedirs("AcccuracyValues/")
        #     open("AcccuracyValues/AcccuracyValues.csv", "x")


        for nNeurons in nNeurons_:
            for nLayers in nLayers_:
                for activationFunc in activationFunc_:
                    for solverFunc in solverFunc_:
                        acc = self.btnStart_Click(nNeurons=nNeurons, nLayers=nLayers, activationFunc=activationFunc, solverFunc=solverFunc)
                        with open("AcccuracyValues/AcccuracyValues.csv", "a") as file:
                            file.write("{}nNeurons_{}nLayers_{}_{}".format(nNeurons, nLayers, a_functions[activationFunc], solvers[solverFunc]) + "," + str(acc) + "\n")
        


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    if GUI:
        ui.setupUi()
        ui.show()
        sys.exit(app.exec_())
    else:
        ui.autorun()