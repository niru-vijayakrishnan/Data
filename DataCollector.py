from sklearn.linear_model import SGDClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from CustomModel import *

class DataCollector:

    #If Custom Model is Used, Put in Class Called CustomModel.py
    #CustomModel.py -> Functions Must Be fit(trainX,trainY), getWeights(), predict(testX)
    
    #Constructor
    def __init__(self,DataFile, penalty, loss="log_loss", regularizationMultiplier=10000,numOfIterations=1000, isCustomModel=False):
        self.DataFile = DataFile
        self.Name = DataFile.replace(".txt", '')
        self.readTextFile()
        self.loss = loss
        self.penalty = penalty
        self.regularizationMultiplier = regularizationMultiplier
        self.numOfIterations = numOfIterations
        self.isCustomModel = isCustomModel

    def readTextFile(self):
        #Separates Data into X and Y from .txt file
        self.X_DATA = []
        self.Y_DATA = []
        with open(self.DataFile) as data_file:
            for line in data_file:
                dataLine = line.split()
                self.Y_DATA.append(float(dataLine.pop()))
                self.X_DATA.append(dataLine)
        for i in range(len(self.X_DATA)):
            for j in range(len(self.X_DATA[i])):
                self.X_DATA[i][j] = float(self.X_DATA[i][j])

    def graphDataWithGivenThresholds(self):
        try:
            hasattr(self.thresholds)
        except:
            self.getWeightThresholds()
        for threshold in self.thresholds:
            self.graphData(threshold)
    
    def getWeightThresholds(self,start=5,stop=40,step=5):
        self.totalWeights = []
        for i in range(1,self.numOfIterations):
            print("Percentage: " + str(i/self.numOfIterations))
            if(self.isCustomModel == False):
                clf = SGDClassifier(loss=self.loss, penalty=self.penalty, alpha = i/self.regularizationMultiplier, max_iter=1000, fit_intercept=True)
                clf.fit(self.X_DATA, self.Y_DATA)
                listOfWeights = [value for value in clf.coef_[0] if value != 0]
            else:
                pass #To Be Implemented
            for weight in listOfWeights:
                self.totalWeights.append(abs(weight))
        self.totalWeights = sorted(self.totalWeights)
        self.thresholds = []
        for i in range(start, stop, step):
            self.thresholds.append(self.totalWeights[int(len(self.totalWeights)*(1-(i*.01)))])

    def graphWeights(self):
        try:
            hasattr(self.totalWeights)
        except:
            self.getWeightThresholds()
        x = []
        for i in range(len(self.totalWeights)):
            x.append(i + 1)
        x = np.array(x)
        y = np.array(self.totalWeights)
        plt.figure()
        plt.scatter(x,y)
        plt.show()

    def collectData(self,threshold,PointDomination="True"):
        self.RawData = []
        for i in range(1,self.numOfIterations):
            print("Percentage: " + str(i/self.numOfIterations))
            self.RawData.append(self.getLogisticRegressionData(i/self.regularizationMultiplier,self.X_DATA, self.Y_DATA, self.X_DATA, self.Y_DATA, threshold))
        optimizedData = self.optimizeForMinimumsOrMaximums(self.RawData)
        formattedData = self.formatData(optimizedData)
        if(PointDomination):
            self.cleanedData = self.doPointDomination(formattedData)
        else:
            self.cleanedData = formattedData

    def graphData(self, threshold):
        try:
            hasattr(self.cleanedData)
        except:
            self.collectData(threshold)
        plt.figure()
        plt.subplot(2,2,1)
        plt.plot(self.cleanedData[0],self.cleanedData[1], label = "Statistical Parity")
        plt.plot(self.cleanedData[0], self.cleanedData[2], label = "Equality of Opportunity")
        plt.plot(self.cleanedData[0], self.cleanedData[3], label = "Accuracy")
        #plt.legend()
        plt.xlabel('Number of Weights')
        plt.ylabel('Statistical Parity')
        plt.title("Minimized for Statistical Parity")
        plt.subplot(2,2,2)
        plt.plot(self.cleanedData[4],self.cleanedData[5], label = "Statistical Parity")
        plt.plot(self.cleanedData[4],self.cleanedData[6], label = "Equality of Opportunity")
        plt.plot(self.cleanedData[4],self.cleanedData[7], label = "Accuracy")
        #plt.legend()
        plt.xlabel('Number of Weights')
        plt.ylabel('Equality of Opportunity')
        plt.title("Minimized for Equality of Opportunity")
        plt.subplot(2,2,3)
        plt.plot(self.cleanedData[8],self.cleanedData[9], label = "Statistical Parity")
        plt.plot(self.cleanedData[8],self.cleanedData[10], label = "Equality of Opportunity")
        plt.plot(self.cleanedData[8],self.cleanedData[11], label = "Accuracy")
        #plt.legend()
        plt.xlabel('Number of Weights')
        plt.ylabel('Accuracy')
        plt.title("Maximized for Accuracy")
        plt.tight_layout()
        plt.savefig(self.Name + "Threshold" + str(threshold) + ".png")
        #plt.show()

    def doPointDomination(self, data):
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

    def formatData(self, data):
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

    def optimizeForMinimumsOrMaximums(self, data):
        cleanedData = []
        weights = self.getUniqueWeights(data)
        for i in range(2):
            for num in weights:
                cleanedData.append(self.getMinimum(data, num, i+1))
        for num in weights:
            cleanedData.append(self.getMaximum(data, num, 3))
        return cleanedData
    
    def getUniqueWeights(self, data):
        diffWeights = []
        for i in range(len(data)):
            diffWeights.append(data[i][4])
        uniqueWeights = np.unique(diffWeights)
        return uniqueWeights.tolist()
    
    def getMaximum(self, data, weight, index):
        maximum = 0
        dataPoint = []
        for i in range(len(data)):
            if data[i][4] == weight:
                if data[i][index] >= maximum:
                    maximum = data[i][index]
                    dataPoint = [index, data[i][4], data[i][1], data[i][2], data[i][3]]
        return dataPoint

    def getMinimum(self, data, weight, index):
        minimum = float('inf')
        dataPoint = []
        for i in range(len(data)):
            if data[i][4] == weight:
                if data[i][index] <= minimum:
                    minimum = data[i][index]
                    dataPoint = [index, data[i][4], data[i][1], data[i][2], data[i][3]]
        return dataPoint

    #Gets Logistic Regression Data
    def getLogisticRegressionData(self, alpha, trainX, trainY, testX, testY, threshold, identifyingFeature=0):
        if(self.isCustomModel == False):
            clf = SGDClassifier(loss=self.loss, penalty=self.penalty, alpha = alpha, max_iter=1000, fit_intercept=True)
            clf.fit(trainX, trainY)
            predictedY = clf.predict(testX)
            numOfWeights = self.countWeights(clf.coef_[0], threshold) #number of non-zero weights
        else:
            pass #To Be Implemented
        statisticalParity = self.getStatisticalParity(identifyingFeature, testX, predictedY)
        #print("Statistical Parity: " + str(statisticalParity))
        equalityOfOpportunity = self.getEqualityOfOpportunity(identifyingFeature, testX, testY, predictedY)
        #print("Equality of Opportunity: " + str(equalityOfOpportunity))
        total = 0
        for i in range(len(testX)):
            if(predictedY[i] == testY[i]):
                total += 1
        accuracy = total/(len(testX))
        #print("Accuracy: " + str(accuracy))
        return [alpha, statisticalParity, equalityOfOpportunity, accuracy, numOfWeights]
    
    def countWeights(self, list, threshold):
        count = 0
        for num in list:
            if abs(num) >= threshold:
                count += 1
        return count

    def getStatisticalParity(self, idenFeature, testX, predictedY):
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
    
    def getEqualityOfOpportunity(self, idenFeature, testX, testY, predictedY):
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