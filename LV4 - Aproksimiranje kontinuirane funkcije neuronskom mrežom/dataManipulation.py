import csv
import numpy as np
import os
import shutil

parameters=[]
MSE=[]

with open("C:\Fakultet\FERIT\8. semestar\Meko računarstvo\LV\LV4\MSEValues\MSEValues.csv", "r") as file:
    csvreader = csv.reader(file)
    for row in csvreader:
        parameters.append(row[0])
        MSE.append(float(row[1]))

parameters = np.asarray(parameters)
MSE = np.asarray(MSE)

indicies = np.argsort(MSE)

parameters = [parameters[i] for i in indicies]
MSE = [MSE[i] for i in indicies]

if not os.path.exists("C:/Fakultet/FERIT/8. semestar/Meko računarstvo/LV/LV4/graphs_final/"):
    os.makedirs("C:/Fakultet/FERIT/8. semestar/Meko računarstvo/LV/LV4/graphs_final/")
    open("C:/Fakultet/FERIT/8. semestar/Meko računarstvo/LV/LV4/graphs_final/MSEValues.txt", "w")

with open("C:/Fakultet/FERIT/8. semestar/Meko računarstvo/LV/LV4/graphs_final/MSEValues.txt", "a") as file:
    file.write("Best values:\n")


for i in range(10):
    src="C:/Fakultet/FERIT/8. semestar/Meko računarstvo/LV/LV4/graphs/" + parameters[i] + ".png"
    dst="C:/Fakultet/FERIT/8. semestar/Meko računarstvo/LV/LV4/graphs_final/" + parameters[i] + ".png"
    with open("C:/Fakultet/FERIT/8. semestar/Meko računarstvo/LV/LV4/graphs_final/MSEValues.txt", "a") as file:
        file.write(parameters[i] + ": " + str(MSE[i]) + "\n")
    shutil.copy(src,dst)


with open("C:/Fakultet/FERIT/8. semestar/Meko računarstvo/LV/LV4/graphs_final/MSEValues.txt", "a") as file:
    file.write("\n\nMedian values:\n")

for i in range(int(len(parameters)/2)-5,int(len(parameters)/2)+5):
    src="C:/Fakultet/FERIT/8. semestar/Meko računarstvo/LV/LV4/graphs/" + parameters[i] + ".png"
    dst="C:/Fakultet/FERIT/8. semestar/Meko računarstvo/LV/LV4/graphs_final/" + parameters[i] + ".png"
    with open("C:/Fakultet/FERIT/8. semestar/Meko računarstvo/LV/LV4/graphs_final/MSEValues.txt", "a") as file:
        file.write(parameters[i] + ": " + str(MSE[i]) + "\n")
    shutil.copy(src,dst)

with open("C:/Fakultet/FERIT/8. semestar/Meko računarstvo/LV/LV4/graphs_final/MSEValues.txt", "a") as file:
    file.write("\n\nWorst values:\n")

for i in range(len(parameters)-10,len(parameters)):
    src="C:/Fakultet/FERIT/8. semestar/Meko računarstvo/LV/LV4/graphs/" + parameters[i] + ".png"
    dst="C:/Fakultet/FERIT/8. semestar/Meko računarstvo/LV/LV4/graphs_final/" + parameters[i] + ".png"
    with open("C:/Fakultet/FERIT/8. semestar/Meko računarstvo/LV/LV4/graphs_final/MSEValues.txt", "a") as file:
        file.write(parameters[i] + ": " + str(MSE[i]) + "\n")
    shutil.copy(src,dst)



