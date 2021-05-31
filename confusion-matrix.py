from array import *
import csv
import random
import math
import operator
import matplotlib.pyplot as plt
import pylab as pl

def loaddataset(filename,training_total):
    with open(filename,'r') as csvfile:
        lines=list(csv.reader(csvfile))
        dataset=list(lines)
        for x in range(1,len(lines)):
            for y in range(len(lines[x])-1):
                lines[x][y]=float(lines[x][y])
            training_total.append(lines[x])


def load_folds(folds, x, training_total, testing=[], training=[]):
    length = len(training_total)
    w = length // folds
    if x == 0:
        for j in range(w):
            testing.append(training_total[j])
        for j in range(w + 1, length):
            training.append(training_total[j])

    elif x == 1:
        for j in range(w + 1, 2 * w):
            testing.append(training_total[j])
        for j in range(w):
            training.append(training_total[j])
        for j in range(2 * w, length):
            training.append(training_total[j])

    elif x == 2:
        for j in range((2 * w) + 1, length):
            testing.append(training_total[j])
        for j in range(2 * w):
            training.append(training_total[j])


def euclidiandistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def get_neighbour(training, testinstance, K):
    distance = []
    length = 8
    for x in range(len(training)):
        dist = euclidiandistance(testinstance, training[x], length)
        distance.append((training[x], dist))
    distance.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(K):
        neighbors.append(distance[x][0])
    return neighbors


def response(neighbors):
    classvotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classvotes:
            classvotes[response] += 1
        else:
            classvotes[response] = 1

    sortedvotes = sorted(classvotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedvotes[0][0]


def getaccuracy(testing,predictions,confusion_matrix):
    correct = 0
    for x in range(len(testing)):
        actualresult= testing[x][len(testing[x])-1]
        predictedresult=predictions[x]

        if(predictedresult==actualresult):
            if(predictedresult=='g'):
                confusion_matrix[1][1]=confusion_matrix[1][1]+1

            else:
                confusion_matrix[0][0] = confusion_matrix[0][0] + 1

        else:
            if(predictedresult=='g'):
                confusion_matrix[1][0] = confusion_matrix[1][0] + 1

            else:
                confusion_matrix[0][1] = confusion_matrix[0][1] + 1




training_total = []
split = 0.65
folds = 3
crossvalid = []
trainingset = []
loaddataset(r'Class_Ionosphere.csv.csv', training_total)
accuracies = {}
confusion_matrix=[[0,0],[0,0]]


for K in range(7,8):

    accuracy_ct = 0
    for m in range(folds):

        crossvalid=[]
        trainingset=[]

        load_folds(folds, m, training_total, crossvalid, trainingset)
        predictions = []

        for y in range(len(crossvalid)):
            neighbors = get_neighbour(trainingset, crossvalid[y], K)
            result = response(neighbors)
            predictions.append(result)

    getaccuracy(crossvalid, predictions,confusion_matrix)

print("Confusion Matrix:")
print(confusion_matrix[0][0], confusion_matrix[0][1])
print(confusion_matrix[1][0], confusion_matrix[1][1])


acc=(confusion_matrix[0][0]+confusion_matrix[1][1])/float(confusion_matrix[0][0]+confusion_matrix[0][1]+confusion_matrix[1][0]+confusion_matrix[1][1])
precision=float(confusion_matrix[0][0])/float(confusion_matrix[0][0]+confusion_matrix[0][1])
recall=float(confusion_matrix[0][0])/float(confusion_matrix[0][0]+confusion_matrix[1][0])
falsepositiverate=float(confusion_matrix[0][1])/float(confusion_matrix[0][1]+confusion_matrix[1][1])
falsenegativerate=float(confusion_matrix[1][0])/float(confusion_matrix[1][0]+confusion_matrix[0][0])
print("Accuracy is ",acc)
print("precison is",precision)
print("recall is",recall)
print("false positive rate is",falsepositiverate)
print("false negative rate is",falsenegativerate)

labels = ['b', 'g']
# cm = confusion_matrix(y_test, pred, labels)
cm = confusion_matrix
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the Class_ionosphere dataset')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

