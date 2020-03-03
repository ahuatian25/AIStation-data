from __future__ import print_function

import keras
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pylab as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedShuffleSplit
from matplotlib.colors import LogNorm
from sklearn.decomposition import PCA
from keras.layers.advanced_activations import PReLU, ELU
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.utils import np_utils
from copy import deepcopy
from keras import metrics
from keras.models import load_model
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder 
from keras.utils import multi_gpu_model
############################################################
#Import data

trainDF=pd.read_csv("E:/XNS/datasets/zj_logg888888.csv")
# trainDF=trainDF[1:50000]
#Clean up wrong X and Y values (very few of them)
xy_scaler=preprocessing.StandardScaler()
xy_scaler.fit(trainDF[["X","Y"]])
trainDF[["X","Y"]]=xy_scaler.transform(trainDF[["X","Y"]])
# print(len(trainDF))
# trainDF=trainDF[abs(trainDF["Y"])<100]
#################Now proceed as before#################
def parse_time(x):
    DD=datetime.strptime(x,"%Y/%m/%d %H:%M")
    time=DD.hour#*60+DD.minute
    day=DD.day
    month=DD.month
    year=DD.year
    return time,day,month,year
#################
def get_season(x):
    summer=0
    fall=0
    winter=0
    spring=0
    if (x in [5, 6, 7]):
        summer=1
    if (x in [8, 9, 10]):
        fall=1
    if (x in [11, 0, 1]):
        winter=1
    if (x in [2, 3, 4]):
        spring=1
    return summer, fall, winter, spring
####################################################
def parse_data(df,logodds,logoddsPA):
    feature_list=df.columns.tolist()
    if "Descript" in feature_list:
        feature_list.remove("Descript")
    if "Resolution" in feature_list:
        feature_list.remove("Resolution")
    if "Category" in feature_list:
        feature_list.remove("Category")
    if "Id" in feature_list:
        feature_list.remove("Id")

    cleanData=df[feature_list]
    cleanData.index=range(len(df))
    print("Creating address features")###Creating address features###
    address_features=cleanData["Address"].apply(lambda x: logodds[x])
    address_features.columns=["logodds"+str(x) for x in range(len(address_features.columns))]
    print("Parsing dates")            ###Creating address features###
    cleanData["Time"], cleanData["Day"], cleanData["Month"], cleanData["Year"]=zip(*cleanData["Dates"].apply(parse_time))
    #     dummy_ranks_DAY = pd.get_dummies(cleanData['DayOfWeek'], prefix='DAY')
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    #     cleanData["DayOfWeek"]=cleanData["DayOfWeek"].apply(lambda x: days.index(x)/float(len(days)))
    print("Creating one-hot variables")
    dummy_ranks_PD = pd.get_dummies(cleanData['PdDistrict'], prefix='PD')
    dummy_ranks_DAY = pd.get_dummies(cleanData["DayOfWeek"], prefix='DAY')
    cleanData["IsInterection"]=cleanData["Address"].apply(lambda x: 1 if "/" in x else 0)
    cleanData["logoddsPA"]=cleanData["Address"].apply(lambda x: logoddsPA[x])
    print("droping processed columns")
    cleanData=cleanData.drop("PdDistrict",axis=1)
    cleanData=cleanData.drop("DayOfWeek",axis=1)
    cleanData=cleanData.drop("Address",axis=1)
    cleanData=cleanData.drop("Dates",axis=1)
    feature_list=cleanData.columns.tolist()
    print("joining one-hot features")
    features = cleanData[feature_list].join(dummy_ranks_PD.ix[:,:]).join(dummy_ranks_DAY.ix[:,:]).join(address_features.ix[:,:])
    print("creating new features")
    features["IsDup"]=pd.Series(features.duplicated()|features.duplicated(keep='last')).apply(int)
    features["Awake"]=features["Time"].apply(lambda x: 1 if (x==0 or (x>=8 and x<=23)) else 0)
    features["Summer"], features["Fall"], features["Winter"], features["Spring"]=zip(*features["Month"].apply(get_season))
    if "Category" in df.columns:
        labels = df["Category"].astype('category')
    #label_names=labels.unique()
    #labels=labels.cat.rename_categories(range(len(label_names)))
    else:
        labels=None
    return features,labels
    #This part is slower than it needs to be.
############################################################################
addresses=sorted(trainDF["Address"].unique())
categories=sorted(trainDF["Category"].unique())
C_counts=trainDF.groupby(["Category"]).size()
A_C_counts=trainDF.groupby(["Address","Category"]).size()
A_counts=trainDF.groupby(["Address"]).size()
logodds={}
logoddsPA={}
MIN_CAT_COUNTS=2
default_logodds=np.log(C_counts/len(trainDF))-np.log(1.0-C_counts/float(len(trainDF)))
for addr in addresses:
    PA=A_counts[addr]/float(len(trainDF))
    logoddsPA[addr]=np.log(PA)-np.log(1.-PA)
    logodds[addr]=deepcopy(default_logodds)
    for cat in A_C_counts[addr].keys():
        if (A_C_counts[addr][cat]>MIN_CAT_COUNTS) and A_C_counts[addr][cat]<A_counts[addr]:
            PA=A_C_counts[addr][cat]/float(A_counts[addr])
            logodds[addr][categories.index(cat)]=np.log(PA)-np.log(1.0-PA)
    logodds[addr]=pd.Series(logodds[addr])
    logodds[addr].index=range(len(categories))
    ###############################################################
trainDF=trainDF.sort_values(by=['Dates'])
features, labels=parse_data(trainDF,logodds,logoddsPA)    
##########################################################################
print(features.columns.tolist())
print(len(features.columns))
##########################################################################
# num_feature_list=["Time","Day","Month","Year","DayOfWeek"]
collist=features.columns.tolist()
scaler = preprocessing.StandardScaler()
scaler.fit(features)
features[collist]=scaler.transform(features)
###########################################################################
##########################################################################################
N_EPOCHS=21
N_HN=128
N_LAYERS=1
N_BATCH=64
DP=0.5
N_CLASS=len(labels.unique())
#########################################################################################
###########################################################################
ros = RandomOverSampler(random_state=0)
featuresArray, labelsArray = ros.fit_resample(features.values,labels.values)
x_train,x_test,y_train,y_test = train_test_split(featuresArray,labelsArray,test_size=0.2,shuffle=False)
# y_train=np_utils.to_categorical(y_train.cat.rename_categories(range(N_CLASS)))
# y_test=np_utils.to_categorical(y_test.cat.rename_categories(range(N_CLASS)))



y_train = keras.utils.to_categorical(LabelEncoder().fit_transform(np.array(y_train)), num_classes=N_CLASS)
y_test = keras.utils.to_categorical(LabelEncoder().fit_transform(np.array(y_test)), num_classes=N_CLASS)
##########################################################################
input_dim=x_train.shape[1]
output_dim=N_CLASS
# Y_train=np_utils.to_categorical(y_train.cat.rename_categories(range(len(y_train.unique()))))

model = Sequential()
model.add(Dense(N_HN,input_dim=input_dim,init='glorot_uniform'))
model.add(BatchNormalization())
model.add(PReLU())
# model.add(Dropout(dp))
for i in range(N_LAYERS):
    model.add(Dense(N_HN, init='glorot_uniform'))
    model.add(BatchNormalization())    
    model.add(PReLU())    
#   model.add(Dropout(dp))
model.add(BatchNormalization())
model.add(Dense(output_dim, init='glorot_uniform'))
model.add(Activation('softmax'))
model = multi_gpu_model(model, 2)
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy', metrics.top_k_categorical_accuracy])

fitting=model.fit(x_train, y_train, epochs=N_EPOCHS, batch_size=N_BATCH,verbose=2,validation_data=(x_test,y_test))
# acc_test, test_score,fitting, model = build_and_fit_model(features_train.values,labels_train,X_test=features_test.values,y_test=labels_test,hn=N_HN,layers=N_LAYERS,epochs=N_EPOCHS,verbose=2,dp=DP)
model.save('jjs_model_0112.h5')

print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
acc_test = model.evaluate(x_test,y_test, batch_size=N_BATCH)
print(acc_test)
print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')