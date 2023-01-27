#Importing necessary packages in Python 
#%matplotlib inline 
import matplotlib.pyplot as plt 

import numpy as np ; np.random.seed(sum(map(ord, "aesthetics")))
import pandas as pd

from sklearn.datasets import make_classification 
#from sklearn.learning_curve import learning_curve 
#from sklearn.cross_validation import train_test_split 
#from sklearn.grid_search import GridSearchCV
#from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import classification_report,confusion_matrix, roc_curve, roc_auc_score, auc, accuracy_score
from sklearn.model_selection import ShuffleSplit,train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize, StandardScaler, MinMaxScaler

import seaborn 
seaborn.set_context('notebook') 
seaborn.set_style(style='darkgrid')

from pprint import pprint 

#https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
file = '../input/germancreditdata/german.data'
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

names = ['existingchecking', 'duration', 'credithistory', 'purpose', 'creditamount', 
         'savings', 'employmentsince', 'installmentrate', 'statussex', 'otherdebtors', 
         'residencesince', 'property', 'age', 'otherinstallmentplans', 'housing', 
         'existingcredits', 'job', 'peopleliable', 'telephone', 'foreignworker', 'classification']

data = pd.read_csv(url,names = names, delimiter=' ')
data.head(10)

# Binarize the y output for easier use of e.g. ROC curves -> 0 = 'bad' credit; 1 = 'good' credit
data.classification.replace([1,2], [1,0], inplace=True)
# Print number of 'good' credits (should be 700) and 'bad credits (should be 300)
data.classification.value_counts()

#numerical variables labels
numvars = ['creditamount', 'duration', 'installmentrate', 'residencesince', 'age', 
           'existingcredits', 'peopleliable', 'classification']

# Standardization
numdata_std = pd.DataFrame(StandardScaler().fit_transform(data[numvars].drop(['classification'], axis=1)))

from collections import defaultdict

#categorical variables labels
catvars = ['existingchecking', 'credithistory', 'purpose', 'savings', 'employmentsince',
           'statussex', 'otherdebtors', 'property', 'otherinstallmentplans', 'housing', 'job', 
           'telephone', 'foreignworker']

d = defaultdict(LabelEncoder)

# Encoding the variable
lecatdata = data[catvars].apply(lambda x: d[x.name].fit_transform(x))

# print transformations
#for x in range(len(catvars)):
    #print(catvars[x],": ", data[catvars[x]].unique())
    #print(catvars[x],": ", lecatdata[catvars[x]].unique())

#One hot encoding, create dummy variables for every category of every categorical variable
dummyvars = pd.get_dummies(data[catvars])

# Unscaled, unnormalized data
#data_clean = pd.concat([data[numvars], dummyvars], axis = 1)

# Scaled, Normalized Data
data_clean = pd.concat([data['classification'], dummyvars, numdata_std], axis = 1)

X_clean = data_clean.drop('classification', axis=1)
y_clean = data_clean['classification']
X_DATA = X_clean.values.tolist()
Y_DATA = y_clean.values.tolist()
#X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(X_clean,y_clean,test_size=0.2, random_state=1)
#X_train_clean.keys()

#print(X_clean)
#print(y_clean)
#print(X_clean.values.tolist())
#print(y_clean.values.tolist())

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

def getZeroes(list):
    count = 0
    for num in list:
        if num >= .1:
            count += 1
    return count

#Gets Logistic Regression Data
def getLogisticRegressionData(alpha, trainX, trainY, testX, testY, identifyingFeature):
    clf = SGDClassifier(loss="log_loss", penalty="l0", alpha = alpha, max_iter=1000, fit_intercept=True)
    clf.fit(trainX, trainY)

    predictedY = clf.predict(testX)

    zeroes = getZeroes(clf.coef_[0])

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

def writeExcel(data):
    df = pd.DataFrame(data, columns = ['Regularizer','Statistical Parity','Equality of Opportunity','Accuracy', 'Number of Weights'])
    df.to_excel('GermanDataL1Loss.xlsx', sheet_name='Data')

#print(X_clean)
#print(X_clean.columns)
#print(X_DATA[0][52])

identifyingIndex = 52

totalData = []
for i in range(1,1000):
    print(i)
    weight = i/10000
    totalData.append(getLogisticRegressionData(weight,X_DATA, Y_DATA, X_DATA, Y_DATA, identifyingIndex))

writeExcel(totalData)
