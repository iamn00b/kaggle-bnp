import numpy as np 
import pandas as pd
import sys
import argparse
import os

parser = argparse.ArgumentParser(description='Preprocess BNP Paribas claim management data')
parser.add_argument('--train', 
                      default=os.path.dirname(os.path.abspath(__file__)) + '/../data_raw/train.csv',
                      help='training data (in csv format)')
parser.add_argument('--test', 
                      default=os.path.dirname(os.path.abspath(__file__)) + '/../data_raw/test.csv',
                      help='testing data (in csv format)')
args = parser.parse_args()

def totalTrueInArray(array):
  count = 0
  for element in array:
    count += 1 if element else 0
  return count

print '== LOADING DATA FILES'
train = pd.read_csv(args.train)
target = train['target'].values
test = pd.read_csv(args.test)
id_test = test['ID'].values

print ' '
print '== DATA INFORMATION'
print 'Train data size (rows, columns) :' , train.shape
print 'Test data size (rows, columns) :' , test.shape
# print 'Trimmed train data size (rows, columns) :' , train.dropna().shape

list_missing_value_rows = train.isnull().any(axis=1)
list_missing_value_columns = train.isnull().any(axis=0)

list_missing_value_rows_count = totalTrueInArray(list_missing_value_rows)
list_missing_value_columns_count = totalTrueInArray(list_missing_value_columns)

print 'Rows count that contains missing value :', list_missing_value_rows_count
print 'Columns count that contains missing value :', list_missing_value_columns_count

print ' '
print '== DROP ID AND TARGET COLUMN'
train = train.drop(['ID', 'target'], axis=1)
test = test.drop(['ID'], axis=1)

print ' '
print '== MISSING VALUE TO MEAN/MODUS'
print 'Fill categorical data with modus in it\'s column'
columnWithString = ["v3", "v22", "v24", "v30", "v31", "v47", "v52", "v56", "v66", "v71", "v74", "v75", "v79", "v91", "v107", "v110", "v112", "v113", "v125"]
for column in columnWithString:
  columnModus = train[column].value_counts().idxmax()
  train[column].fillna(columnModus)
  test[column].fillna(columnModus)

print 'Fill numerical data with mean in it\'s column'
columnMean = train.mean()
train = train.fillna(columnMean)
test = test.fillna(columnMean)

print ' '
print '== CONVERT CATEGORICAL INTO ONE HOT ENCODING'
preprocess_table = pd.concat([train,test], axis=0)
print 'Encode data into one-hot encoding'
data_encoded = pd.get_dummies(preprocess_table, prefix=column)
train_rows = train.shape[0]
print 'Split encoding to train and test'
train = data_encoded.iloc[:train_rows, :]
test = data_encoded.iloc[train_rows:, :] 

print train[:2]
print test[:2]