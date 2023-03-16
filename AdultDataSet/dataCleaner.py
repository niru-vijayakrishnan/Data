import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix, roc_curve, roc_auc_score, auc, accuracy_score
from sklearn.model_selection import ShuffleSplit,train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize, StandardScaler, MinMaxScaler
import numpy as np

#https://archive.ics.uci.edu/ml/datasets/adult
file = '../input/germancreditdata/german.data'
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
identifyingIndex = 58

names = ['age', 'workclass', 'fnlwgt', 'education', 'education.num', 'marital.status',
        'occupation', 'relationship', 'race', 'sex', 'capital.gain', 'capital.loss', 'hours.per.week',
        'native.country', 'income']

data = pd.read_csv(url,names = names, delimiter=' ')
for name in names:
    data[name] = data[name].str.replace(',','')

# Binarize the y output for easier use of e.g. ROC curves -> 0 = 'bad' credit; 1 = 'good' credit
data.income.replace(['<=50K','>50K'], [1,0], inplace=True)
# Print number of 'good' credits (should be 700) and 'bad credits (should be 300)
data.income.value_counts()

#numerical variables labels
numvars = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week', 'income']

numdata_std = pd.DataFrame(StandardScaler().fit_transform(data[numvars].drop(['income'], axis=1)))

from collections import defaultdict

#categorical variables labels
catvars = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']

d = defaultdict(LabelEncoder)

# Encoding the variable
lecatdata = data[catvars].apply(lambda x: d[x.name].fit_transform(x))

#One hot encoding, create dummy variables for every category of every categorical variable
dummyvars = pd.get_dummies(data[catvars])

# Scaled, Normalized Data
data_clean = pd.concat([data['income'], dummyvars, numdata_std], axis = 1)
variable = list(data_clean.columns.values)
for i in range(len(variable)):
    print(str(i-1) + ": " + str(variable[i]))

X_clean = data_clean.drop('income', axis=1)
y_clean = data_clean['income']
X_DATA = X_clean.values.tolist()
Y_DATA = y_clean.values.tolist()

for i in range(len(X_DATA)):
    X_DATA[i].append(Y_DATA[i])

#Move Classifying Index to [0]
for i in range(len(X_DATA)):
    X_DATA.insert(0,X_DATA.pop(identifyingIndex))

myFile = open("AdultData.txt", 'r+')
myArray = np.array(X_DATA)
np.savetxt(myFile, myArray)
myFile.close()