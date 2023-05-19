from sklearn.linear_model import SGDClassifier

class DataCollector:
    
    #Constructor
    def __init__(self,DataFile, loss):
        self.DataFile = DataFile
        self.Name = DataFile.replace(".txt", '')
        self.RawData = []
        self.CleanedData = []
        self.loss = loss

    #Collects Raw Data
    def collectRawData(self):
        identifyingIndex = 0 #Feature that is tested

        #Separates Data into X and Y from .txt file
        X_DATA = []
        Y_DATA = []
        with open(self.DataFile) as data_file:
            for line in data_file:
                dataLine = line.split()
                Y_DATA.append(float(dataLine.pop()))
                X_DATA.append(dataLine)
        for i in range(len(X_DATA)):
            for j in range(len(X_DATA[i])):
                X_DATA[i][j] = float(X_DATA[i][j])

        #Starts Collecting Data
        totalData = []
        for i in range(1,1000):
            print(i)
            weight = i/10000
            totalData.append(getLogisticRegressionData(weight,X_DATA, Y_DATA, X_DATA, Y_DATA, identifyingIndex))
        
        #Writes Data Collection into RawData
        self.RawData = totalData

    def collectCleanedData(self):
        pass

    def graphCleanedData(self):
        pass

    def writeDataToExcel(self):
        pass

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
    
    #Gets Logistic Regression Data
    def getLogisticRegressionData(self, alpha, trainX, trainY, testX, testY, identifyingFeature):
        clf = SGDClassifier(loss="log_loss", penalty=self.loss, alpha = alpha, max_iter=1000, fit_intercept=True)
        clf.fit(trainX, trainY)

        predictedY = clf.predict(testX)

        zeroes = getZeroes(clf.coef_[0]) #number of non-zero weights

        statisticalParity = self.getStatisticalParity(identifyingFeature, testX, predictedY)
        #print("Statistical Parity: " + str(statisticalParity))

        equalityOfOpportunity = self.getEqualityOfOpportunity(identifyingFeature, testX, testY, predictedY)
        
        #print("Equality of Opportunity: " + str(equalityOfOpportunity))
