import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

loss = "l1"

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
def getZeroes(list):
    count = 0
    for num in list:
        if abs(num) >= .1:
            count += 1
    return count

#Gets Logistic Regression Data
def getLogisticRegressionData(alpha, trainX, trainY, testX, testY, identifyingFeature):
    clf = SGDClassifier(loss="log_loss", penalty=loss, alpha = alpha, max_iter=1000, fit_intercept=True)
    clf.fit(trainX, trainY)

    predictedY = clf.predict(testX)

    zeroes = getZeroes(clf.coef_[0]) #number of non-zero weights

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

    return [alpha, statisticalParity, equalityOfOpportunity, accuracy, zeroes]

#DATA COLLECTION COMPLETE

#WRITING TO EXCEL
def getUniqueWeights(data):
    diffWeights = []
    for i in range(len(data)):
        diffWeights.append(data[i][4])
    uniqueWeights = np.unique(diffWeights)
    return uniqueWeights.tolist()

def getMinimum(data, weight, index):
    minimum = 0
    dataPoint = []
    for i in range(len(data)):
        if data[i][4] == weight:
            if data[i][index] >= minimum:
                minimum = data[i][index]
                dataPoint = [index, data[i][4], data[i][1], data[i][2], data[i][3]]
    return dataPoint

def getUniques(list, index):
    diffUniques = []
    for i in range(len(list)):
        diffUniques.append(list[i][index])
    uniques = np.unique(diffUniques)
    return uniques.tolist()

def formatData(data):
    statisticalParityX = []
    statisticalParityY = []
    equalityOfOpportunityX = []
    equalityOfOpportunityY = []
    accuracyX = []
    accuracyY = []
    for i in range(len(data)):
        if(data[i][0] == 1):
            statisticalParityX.append(data[i][1])
            statisticalParityY.append(data[i][2])
        elif(data[i][0] == 2):
            equalityOfOpportunityX.append(data[i][1])
            equalityOfOpportunityY.append(data[i][3])
        elif(data[i][0] == 3):
            accuracyX.append(data[i][1])
            accuracyY.append(data[i][4])
    return [statisticalParityX, statisticalParityY, equalityOfOpportunityX, equalityOfOpportunityY, accuracyX, accuracyY]

def getMinimums(data):
    cleanedData = []
    weights = getUniqueWeights(data)
    for i in range(3):
        for num in weights:
            cleanedData.append(getMinimum(data, num, i+1))
    return cleanedData

def writeExcel(data):
    df = pd.DataFrame(data, columns = ['Regularizer','Statistical Parity','Equality of Opportunity','Accuracy', 'Number of Weights'])
    df.to_excel("Raw" + DataName + loss + "Loss.xlsx", sheet_name='Data')
    minimums = getMinimums(data)
    df = pd.DataFrame(minimums, columns = ['Measurement Type','Number of Weights','Statistical Parity','Equality of Opportunity', 'Accuracy'])
    df.to_excel("Cleaned" + DataName + loss + "Loss.xlsx", sheet_name='Data')

def graph(data):
    minimums = getMinimums(data)
    rdyToGraph = formatData(minimums)
    plt.plot(rdyToGraph[0],rdyToGraph[1])
    plt.xlabel('Number of Weights')
    plt.ylabel('Statistical Parity')
    plt.title("Number of Weights vs Statistical Parity")
    plt.show()
    plt.plot(rdyToGraph[2],rdyToGraph[3])
    plt.xlabel('Number of Weights')
    plt.ylabel('Equality of Opportunity')
    plt.title("Number of Weights vs Equality of Opportunity")
    plt.show()
    plt.plot(rdyToGraph[4],rdyToGraph[5])
    plt.xlabel('Number of Weights')
    plt.ylabel('Accuracy')
    plt.title("Number of Weights vs Accuracy")
    plt.show()

def graphData(DataFile):
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
        print(i)
        weight = i/10000
        totalData.append(getLogisticRegressionData(weight,X_DATA, Y_DATA, X_DATA, Y_DATA, identifyingIndex))

    graph(totalData)
    #writeExcel(totalData)

graphData("GermanData.txt")

