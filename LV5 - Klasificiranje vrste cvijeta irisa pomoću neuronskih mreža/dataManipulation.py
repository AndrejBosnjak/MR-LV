import csv
import numpy as np
import os
import shutil

parameters=[]
Accuracy=[]

parentFile="C:\Fakultet\FERIT\8. semestar\Meko računarstvo\LV\LV5"

with open(parentFile + "\AcccuracyValues\AcccuracyValues.csv", "r") as file:
    csvreader = csv.reader(file)
    for row in csvreader:
        parameters.append(row[0])
        Accuracy.append(float(row[1]))

parameters = np.asarray(parameters)
Accuracy = np.asarray(Accuracy)

indicies = np.argsort(Accuracy)

parameters = [parameters[i] for i in indicies]
Accuracy = [Accuracy[i] for i in indicies]

if not os.path.exists(parentFile + "\graphs_final"):
    os.makedirs(parentFile + "\graphs_final")
    open(parentFile + "\graphs_final\AccuracyValues.txt", "w")

with open(parentFile + "\graphs_final\AccuracyValues.txt", "a") as file:
    file.write("Best values:\n")


for i in range(len(parameters)-10,len(parameters)):
    src=parentFile + "/graphs/" + parameters[i] + ".png"
    dst=parentFile + "/graphs_final/" + parameters[i] + ".png"
    with open(parentFile + "\graphs_final\AccuracyValues.txt", "a") as file:
        file.write(parameters[i] + ": " + str(Accuracy[i]) + "\n")
    shutil.copy(src,dst)


with open(parentFile + "\graphs_final\AccuracyValues.txt", "a") as file:
    file.write("\n\nMedian values:\n")

for i in range(int(len(parameters)/2)-5,int(len(parameters)/2)+5):
    src=parentFile + "/graphs/" + parameters[i] + ".png"
    dst=parentFile + "/graphs_final/" + parameters[i] + ".png"
    with open(parentFile + "\graphs_final\AccuracyValues.txt", "a") as file:
        file.write(parameters[i] + ": " + str(Accuracy[i]) + "\n")
    shutil.copy(src,dst)

with open(parentFile + "\graphs_final\AccuracyValues.txt", "a") as file:
    file.write("\n\nWorst values:\n")

for i in range(10):
    src=parentFile + "/graphs/" + parameters[i] + ".png"
    dst=parentFile + "/graphs_final/" + parameters[i] + ".png"
    with open(parentFile + "\graphs_final\AccuracyValues.txt", "a") as file:
        file.write(parameters[i] + ": " + str(Accuracy[i]) + "\n")
    shutil.copy(src,dst)



