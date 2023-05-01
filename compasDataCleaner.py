import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix, roc_curve, roc_auc_score, auc, accuracy_score
from sklearn.model_selection import ShuffleSplit,train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize, StandardScaler, MinMaxScaler
import numpy as np

data = pd.read_csv('cox-violent-parsed_filt.csv',header = 0, delimiter=',')

#numerical variables labels
numvars = ['age', 'juv_fel_count', 'decile_score', 'juv_misd_count',
           'juv_other_count', 'priors_count',
           'is_recid', 'is_violent_recid', 'decile_score', 'v_decile_score', 'priors_count',
           'event']

numdata_std = pd.DataFrame(StandardScaler().fit_transform(data[numvars].drop(['event'], axis=1)))

from collections import defaultdict

#categorical variables labels
catvars = ['sex','race', 'type_of_assessment', 'score_text',
           'v_type_of_assessment', 'v_score_text']

d = defaultdict(LabelEncoder)

#print(data)
#dataToDrop = ['name','first','last','dob','c_jail_in', 'c_jail_out','r_offense_date','r_jail_in','vr_offense_date','screening_date','id','days_b_screening_arrest','c_days_from_compas','r_days_from_arrest']

'''
data = data.drop(columns=dataToDrop)
for data in dataToDrop:
    try:
        numvars.remove(data)
    except:
        pass
    try:
        catvars.remove(data)
    except:
        pass
#droppedData = ['name','first']
#data = data.drop(droppedData, axis = 1)
#print(data)
'''

# Encoding the variable
lecatdata = data[catvars].apply(lambda x: d[x.name].fit_transform(x))

#One hot encoding, create dummy variables for every category of every categorical variable
dummyvars = pd.get_dummies(data[catvars])

# Scaled, Normalized Data
data_clean = pd.concat([data['event'], dummyvars, numdata_std], axis = 1)
variable = list(data_clean.columns.values)
for i in range(len(variable)):
    print(str(i-1) + ": " + str(variable[i]))

X_clean = data_clean.drop('event', axis=1)
y_clean = data_clean['event']
X_DATA = X_clean.values.tolist()
Y_DATA = y_clean.values.tolist()


#.txt file formatting
for i in range(len(X_DATA)):
    X_DATA[i].append(Y_DATA[i])
myFile = open("CompasData.txt", 'r+')
myArray = np.array(X_DATA)
np.savetxt(myFile, myArray)
myFile.close()