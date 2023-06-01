import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pp

#To Change Between Datasets
#DataFile = "AdultData.txt"
#loss = "l1"
#threshold = 0.1

def getStatisticalParity(idenFeature, testX, predictedY):
    group1X = []
    group1Y = []
    group2X = []
    group2Y = []
    #Separates Groups
    for i in range(len(testX)):
        if testX[i][idenFeature] == 0:
            group1X.append(testX[i])
            group1Y.append(predictedY[i])
        else:
            group2X.append(testX[i])
            group2Y.append(predictedY[i])
    #Computes Fraction of Group 1
    numGroup1X = 0
    for i in range(len(group1Y)):
        if group1Y[i] == 1:
            numGroup1X += 1
    fractionGroup1X = float(numGroup1X/len(group1Y))
    #Computes Fraction of Group 2
    numGroup2X = 0
    for i in range(len(group2Y)):
        if group2Y[i] == 1:
            numGroup2X += 1
    fractionGroup2X = float(numGroup2X/len(group2Y))
    #Finds Absolute Value of the Difference
    return abs(fractionGroup1X-fractionGroup2X)

def getEqualityOfOpportunity(idenFeature, testX, testY, predictedY):
    group1X = []
    group1Y = []
    group2X = []
    group2Y = []
    #Separates Groups
    for i in range(len(testX)):
        if testY[i] == 1:
            if testX[i][idenFeature] == 0:
                group1X.append(testX[i])
                group1Y.append(predictedY[i])
            else:
                group2X.append(testX[i])
                group2Y.append(predictedY[i])
    #Computes Fraction of Group 1
    numGroup1X = 0
    for i in range(len(group1Y)):
        if group1Y[i] == 0.0:
            numGroup1X += 1
    fractionGroup1X = float(numGroup1X/len(group1Y))
    #Computes Fraction of Group 2
    numGroup2X = 0
    for i in range(len(group2Y)):
        if group2Y[i] == 0.0:
            numGroup2X += 1
    fractionGroup2X = float(numGroup2X/len(group2Y))
    #Finds Absolute Value of the Difference
    return abs(fractionGroup1X-fractionGroup2X)

from sklearn.linear_model import SGDClassifier

#Not Zeroes, Actually number of non-zero weights
def getZeroes(list, threshold):
    count = 0
    for num in list:
        if abs(num) >= threshold:
            count += 1
    return count

#Gets Logistic Regression Data
def getLogisticRegressionData(alpha, trainX, trainY, testX, testY, identifyingFeature, threshold, lossPenalty):
    clf = SGDClassifier(loss="log_loss", penalty=lossPenalty, alpha = alpha, max_iter=1000, fit_intercept=True)
    clf.fit(trainX, trainY)

    predictedY = clf.predict(testX)

    listOfWeights = [value for value in clf.coef_[0] if value != 0]
    #print(listOfWeights)
    zeroes = getZeroes(clf.coef_[0], threshold) #number of non-zero weights

    statisticalParity = getStatisticalParity(identifyingFeature, testX, predictedY)
    #print("Statistical Parity: " + str(statisticalParity))

    equalityOfOpportunity = getEqualityOfOpportunity(identifyingFeature, testX, testY, predictedY)
    
    #print("Equality of Opportunity: " + str(equalityOfOpportunity))


    total = 0
    for i in range(len(testX)):
        if(predictedY[i] == testY[i]):
            total += 1
    accuracy = total/(len(testX))
    #print("Accuracy: " + str(accuracy))

    return [alpha, statisticalParity, equalityOfOpportunity, accuracy, zeroes, listOfWeights]

#DATA COLLECTION COMPLETE

#WRITING TO EXCEL
def getUniqueWeights(data):
    diffWeights = []
    for i in range(len(data)):
        diffWeights.append(data[i][4])
    uniqueWeights = np.unique(diffWeights)
    return uniqueWeights.tolist()

def getUniques(list, index):
    diffUniques = []
    for i in range(len(list)):
        diffUniques.append(list[i][index])
    uniques = np.unique(diffUniques)
    return uniques.tolist()

def formatData(data):
    statisticalParityX = []
    statisticalParityY = []
    SPEOY = []
    SPAY = []
    equalityOfOpportunityX = []
    equalityOfOpportunityY = []
    EOSPY = []
    EOAY = []
    accuracyX = []
    accuracyY = []
    ASPY = []
    AEOY = []
    for i in range(len(data)):
        if(data[i][0] == 1):
            statisticalParityX.append(data[i][1])
            statisticalParityY.append(data[i][2])
            SPEOY.append(data[i][3])
            SPAY.append(data[i][4])
        elif(data[i][0] == 2):
            equalityOfOpportunityX.append(data[i][1])
            equalityOfOpportunityY.append(data[i][3])
            EOSPY.append(data[i][2])
            EOAY.append(data[i][4])
        elif(data[i][0] == 3):
            accuracyX.append(data[i][1])
            accuracyY.append(data[i][4])
            ASPY.append(data[i][2])
            AEOY.append(data[i][3])
    return [statisticalParityX, statisticalParityY, SPEOY, SPAY, equalityOfOpportunityX, EOSPY, equalityOfOpportunityY, EOAY, accuracyX, ASPY, AEOY, accuracyY]

def doPointDomination(data):
    statisticalParityX = data[0]
    statisticalParityY = data[1]
    SPEOY = data[2]
    SPAY = data[3]
    equalityOfOpportunityX = data[4]
    equalityOfOpportunityY = data[6]
    EOSPY = data[5]
    EOAY = data[7]
    accuracyX = data[8]
    accuracyY = data[11]
    ASPY = data[9]
    AEOY = data[10]
    newStatisticalParityX = []
    newStatisticalParityY = []
    newSPEOY = []
    newSPAY = []
    newEqualityOfOpportunityX = []
    newEqualityOfOpportunityY = []
    newEOSPY = []
    newEOAY = []
    newAccuracyX = []
    newAccuracyY = []
    newASPY = []
    newAEOY = []
    maxStatisticalParity = float('-inf')
    for i in range(len(statisticalParityY)):
        if statisticalParityY[i] >= maxStatisticalParity:
            newStatisticalParityX.append(statisticalParityX[i])
            newStatisticalParityY.append(statisticalParityY[i])
            newSPEOY.append(SPEOY[i])
            newSPAY.append(SPAY[i])
            maxStatisticalParity = statisticalParityY[i]
    maxEqualityOfOpportunity = float('-inf')
    for i in range(len(equalityOfOpportunityY)):
        if equalityOfOpportunityY[i] >= maxEqualityOfOpportunity:
            newEqualityOfOpportunityX.append(equalityOfOpportunityX[i])
            newEqualityOfOpportunityY.append(equalityOfOpportunityY[i])
            newEOSPY.append(EOSPY[i])
            newEOAY.append(EOAY[i])
            maxEqualityOfOpportunity = equalityOfOpportunityY[i]
    maxAccuracy = float('-inf')
    for i in range(len(accuracyY)):
        if accuracyY[i] >= maxAccuracy:
            newAccuracyX.append(accuracyX[i])
            newAccuracyY.append(accuracyY[i])
            newASPY.append(ASPY[i])
            newAEOY.append(AEOY[i])
            maxAccuracy = accuracyY[i]
    return [newStatisticalParityX, newStatisticalParityY, newSPEOY, newSPAY, newEqualityOfOpportunityX, newEOSPY, newEqualityOfOpportunityY, newEOAY, newAccuracyX, newASPY, newAEOY, newAccuracyY]

def getMinimumsOrMaximums(data):
    #pp(data)
    cleanedData = []
    weights = getUniqueWeights(data)
    for i in range(2):
        for num in weights:
            cleanedData.append(getMinimum(data, num, i+1))
    for num in weights:
        cleanedData.append(getMaximum(data, num, 3))
    return cleanedData

def getMaximum(data, weight, index):
    maximum = 0
    dataPoint = []
    for i in range(len(data)):
        if data[i][4] == weight:
            if data[i][index] >= maximum:
                maximum = data[i][index]
                dataPoint = [index, data[i][4], data[i][1], data[i][2], data[i][3]]
    return dataPoint

def getMinimum(data, weight, index):
    minimum = float('inf')
    dataPoint = []
    for i in range(len(data)):
        if data[i][4] == weight:
            if data[i][index] <= minimum:
                minimum = data[i][index]
                dataPoint = [index, data[i][4], data[i][1], data[i][2], data[i][3]]
    return dataPoint

def writeExcel(data, DataName, loss):
    df = pd.DataFrame(data, columns = ['Regularizer','Statistical Parity','Equality of Opportunity','Accuracy', 'Number of Weights'])
    df.to_excel("Raw" + DataName + loss + "Loss.xlsx", sheet_name='Data')
    minimums = getMinimumsOrMaximums(data)
    df = pd.DataFrame(minimums, columns = ['Measurement Type','Number of Weights','Statistical Parity','Equality of Opportunity', 'Accuracy'])
    df.to_excel("Cleaned" + DataName + loss + "Loss.xlsx", sheet_name='Data')

def graph(data, DataName, threshold):
    minimums = getMinimumsOrMaximums(data)
    organizedData = formatData(minimums)
    rdyToGraph = doPointDomination(organizedData)
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(rdyToGraph[0],rdyToGraph[1], label = "Statistical Parity")
    plt.plot(rdyToGraph[0], rdyToGraph[2], label = "Equality of Opportunity")
    plt.plot(rdyToGraph[0], rdyToGraph[3], label = "Accuracy")
    #plt.legend()
    plt.xlabel('Number of Weights')
    plt.ylabel('Statistical Parity')
    plt.title("Minimized for Statistical Parity")
    plt.subplot(2,2,2)
    plt.plot(rdyToGraph[4],rdyToGraph[5], label = "Statistical Parity")
    plt.plot(rdyToGraph[4],rdyToGraph[6], label = "Equality of Opportunity")
    plt.plot(rdyToGraph[4],rdyToGraph[7], label = "Accuracy")
    #plt.legend()
    plt.xlabel('Number of Weights')
    plt.ylabel('Equality of Opportunity')
    plt.title("Minimized for Equality of Opportunity")
    plt.subplot(2,2,3)
    plt.plot(rdyToGraph[8],rdyToGraph[9], label = "Statistical Parity")
    plt.plot(rdyToGraph[8],rdyToGraph[10], label = "Equality of Opportunity")
    plt.plot(rdyToGraph[8],rdyToGraph[11], label = "Accuracy")
    #plt.legend()
    plt.xlabel('Number of Weights')
    plt.ylabel('Accuracy')
    plt.title("Maximized for Accuracy")
    plt.tight_layout()
    plt.savefig(DataName + "Threshold" + str(threshold) + ".png")
    #plt.show()

def graphWeights(list):
    x = []
    for i in range(len(list)):
        x.append(i + 1)
    x = np.array(x)
    y = np.array(list)
    plt.scatter(x,y)
    plt.show()

def graphData(DataFile, threshold, loss):
    identifyingIndex = 0
    DataName = DataFile.replace(".txt", '')
    X_DATA = []
    Y_DATA = []
    with open(DataFile) as data_file:
        for line in data_file:
            dataLine = line.split()
            Y_DATA.append(float(dataLine.pop()))
            X_DATA.append(dataLine)
    for i in range(len(X_DATA)):
        for j in range(len(X_DATA[i])):
            X_DATA[i][j] = float(X_DATA[i][j])
    totalData = []
    for i in range(1,1000):
        if i % 10 == 0:
            print(str(int(i/10)) + "%")
        weight = i/10000
        totalData.append(getLogisticRegressionData(weight,X_DATA, Y_DATA, X_DATA, Y_DATA, identifyingIndex, threshold, loss))
    totalWeights = []
    for row in totalData:
        if row:  # Ensure the row is not empty
            last_element = row.pop()
            for num in last_element:
                totalWeights.append(abs(num))
    totalWeights = sorted(totalWeights)
    #graphWeights(totalWeights)
    #print(totalWeights)
    graph(totalData, DataName, threshold)
    #writeExcel(totalData, loss)

for i in range(1,10):
    graphData("CompasData.txt", threshold = i*.2, loss = "l2")
