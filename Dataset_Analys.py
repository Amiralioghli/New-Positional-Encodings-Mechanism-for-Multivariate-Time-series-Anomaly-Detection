
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import VarianceThreshold
import seaborn as sns

from sklearn.feature_selection import mutual_info_classif


#train1 = pd.read_csv("//Users//macbookpro//Documents//TSAD//Datasets//HAI_Binary_class//haiend-23.05//end-test1.csv")
train2 = pd.read_csv("//Users//macbookpro//Documents//TSAD//Datasets//HAI_Binary_class//haiend-23.05//end-test2.csv")

train2 = train2.drop(["Timestamp"], axis = 1)

train2_label = pd.read_csv("//Users//macbookpro//Documents//TSAD//Datasets//HAI_Binary_class//haiend-23.05//label-test2.csv")

train2_label = pd.DataFrame(train2_label["label"])

undersampling = RandomUnderSampler(sampling_strategy='majority')
#sm = SMOTE(random_state = 2) 


X_train_res, y_train_res = undersampling.fit_resample(train2, train2_label) 

print("Before undersampling: ", train2_label["label"].value_counts())
print("After undersampling: ", y_train_res["label"].value_counts())


train_csv_path=pd.read_csv('//Users//macbookpro//Documents//TSAD//Datasets//HAI_Binary_class//train_feature_selected1.csv')
test_csv_path=pd.read_csv('//Users//macbookpro//Documents//TSAD//Datasets//HAI_Binary_class//test_feature_selected1.csv')


# 2836 + 16070 = 18906
# Before sampling
# 0    221997
# 1      8403

# After sampling 
# 0     8905 + 1598 = 10503
# 1     7165 + 1238 = 8403

print("After undersampling: ", train_csv_path["label"].value_counts())
print("After undersampling: ", test_csv_path["label"].value_counts())


full_train_dataset = pd.concat([X_train_res , y_train_res], axis=1)

print("Before undersampling: ", full_train_dataset["label"].value_counts())
print("After undersampling: ", full_train_dataset["label"].value_counts())


#  1. preprocessing first step (VarianceThreshold)

var_threshold = VarianceThreshold(threshold=0)

var_threshold.fit(full_train_dataset)

constant_columns1 = [column for column in full_train_dataset.columns
                    if column not in full_train_dataset.columns[var_threshold.get_support()]]
print(len(constant_columns1))

# after applying VarianceThreshold 151 feature removed

full_train_dataset_v1 = full_train_dataset.drop(constant_columns1,axis=1)


#  2. preprocessing Second step (With Correlation)

full_train_dataset_v1 = pd.DataFrame(full_train_dataset_v1.values)
#Using Pearson Correlation
plt.figure(figsize=(12,10))
cor = full_train_dataset_v1.iloc[:,:10].corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
plt.show()


def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr

corr_features = correlation(full_train_dataset_v1, 0.90)
len(set(corr_features))

# after applying Correlation feature enginnering 37 feature removed


full_train_dataset_v2 = full_train_dataset_v1.drop(corr_features,axis=1)


#  2. preprocessing Second step (With Correlation)

labels = full_train_dataset_v2.iloc[:,-1]

input = full_train_dataset_v2.drop([74], axis=1)


mutual_info = mutual_info_classif(input, labels)

len(mutual_info)
input = pd.DataFrame(input.values())
mutual_info = pd.Series(mutual_info)
mutual_info.index = input.columns
mutual_info.sort_values(ascending=False)

mutual_info.sort_values(ascending=False).plot.bar(figsize=(20, 8))

mutual_info.sort_values(ascending=False).plot.bar(figsize=(20, 8))
plt.xlabel('Features name (total 38 feature)', fontsize=12)
plt.ylabel('The Features Information Gain (Score)', fontsize=12)
plt.show()





