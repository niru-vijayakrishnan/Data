import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix, roc_curve, roc_auc_score, auc, accuracy_score
from sklearn.model_selection import ShuffleSplit,train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize, StandardScaler, MinMaxScaler
import numpy as np


#https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
file = '../input/germancreditdata/german.data'
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
identifyingIndex = 52 #German Worker

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
variable = list(data_clean.columns.values)
for i in range(len(variable)):
    print(str(i-1) + ": " + str(variable[i]))

X_clean = data_clean.drop('classification', axis=1)
y_clean = data_clean['classification']
X_DATA = X_clean.values.tolist()
Y_DATA = y_clean.values.tolist()
#X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(X_clean,y_clean,test_size=0.2, random_state=1)
#X_train_clean.keys()

for i in range(len(X_DATA)):
    X_DATA[i].append(Y_DATA[i])

#Move Classifying Index to [0]
for i in range(len(X_DATA)):
    X_DATA.insert(0,X_DATA.pop(identifyingIndex))
    
myFile = open('GermanData.txt', 'r+')
myArray = np.array(X_DATA)
np.savetxt(myFile, myArray)
myFile.close()